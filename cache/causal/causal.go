package causal

import (
	"errors"
	"fmt"
	"log/slog"
	"math"
	"slices"

	"github.com/ollama/ollama/cache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
)

// TODO(jessegross): This needs to have unit tests

type Causal struct {
	DType    ml.DType
	Capacity int32

	// ** current forward pass **

	// the active layer for Get and Put
	curLayer int

	// starting location for data storage for this batch
	curLoc int

	// size of the current batch
	curBatchSize int

	// mask of the cache as used by this batch
	curMask ml.Tensor

	// locations in the cache that are needed for this batch
	curCellRange cellRange

	// ** cache metadata **

	// for each possible location in the cache, stores the position and set of sequences
	// that reference the data there
	cells []cacheCell

	// maps from sequence to the range of locations where it is stored in the cache
	cellRanges map[int]cellRange

	// ** cache data storage **

	model        model.Model
	cacheCtx     ml.Context
	keys, values []ml.Tensor
}

type cacheCell struct {
	pos       int32
	sequences []int
}

type cellRange struct {
	min int
	max int
}

func NewCausalCache(model model.Model, dtype ml.DType, capacity int32) cache.Cache {
	return &Causal{
		Capacity:   capacity,
		DType:      dtype,
		cells:      make([]cacheCell, capacity),
		cellRanges: make(map[int]cellRange),
		model:      model,
		cacheCtx:   model.Backend().NewContext(),
	}
}

func (c *Causal) Close() {
	c.cacheCtx.Close()
}

func (c *Causal) StartForward(ctx ml.Context, positions []int32, seqs []int) error {
	if len(positions) != len(seqs) {
		return fmt.Errorf("length of positions (%v) must match length of seqs (%v)", len(positions), len(seqs))
	}

	c.curBatchSize = len(positions)

	if c.curBatchSize < 1 {
		return errors.New("batch size cannot be less than 1")
	}

	var err error
	c.curLoc, err = c.findStartLoc()
	if errors.Is(err, cache.ErrKvCacheFull) {
		c.defrag()
		c.curLoc, err = c.findStartLoc()
	}
	if err != nil {
		return err
	}

	c.curCellRange = newRange()
	for i, pos := range positions {
		seq := seqs[i]

		c.cells[c.curLoc+i] = cacheCell{pos: pos, sequences: []int{seq}}

		seqRange, ok := c.cellRanges[seq]
		if !ok {
			seqRange = newRange()
		}

		if c.curLoc+i > seqRange.max {
			seqRange.max = c.curLoc + i
		}
		if seqRange.max > c.curCellRange.max {
			c.curCellRange.max = seqRange.max
		}

		if c.curLoc+i < seqRange.min {
			seqRange.min = c.curLoc + i
		}
		if seqRange.min < c.curCellRange.min {
			c.curCellRange.min = seqRange.min
		}
		c.cellRanges[seq] = seqRange
	}

	c.curMask, err = c.buildMask(ctx, positions, seqs)

	return err
}

func newRange() cellRange {
	return cellRange{
		min: math.MaxInt,
		max: 0,
	}
}

// Find the first contiguous block of at least curBatchSize
func (c *Causal) findStartLoc() (int, error) {
	var start, count int
	for i := range c.cells {
		if len(c.cells[i].sequences) == 0 {
			count++
			if count >= c.curBatchSize {
				return start, nil
			}
		} else {
			start = i + 1
			count = 0
		}
	}

	return 0, fmt.Errorf("%w (length: %v)", cache.ErrKvCacheFull, c.Capacity)
}

// Builds a mask of history x batch indicating whether for each token in the batch the
// token in the history should apply. This is based on both the sequence and causality (the
// position of the history is not ahead of the token in the batch).
func (c *Causal) buildMask(ctx ml.Context, positions []int32, seqs []int) (ml.Tensor, error) {
	// TODO(jessegross): This makes a number of simplifications such as no padding,
	// which could be an issue for CUDA graphs and/or flash attention
	len := c.curCellRange.max - c.curCellRange.min + 1
	mask := make([]float32, c.curBatchSize*len)

	for i := range c.curBatchSize {
		for j := c.curCellRange.min; j <= c.curCellRange.max; j++ {
			if !slices.Contains(c.cells[j].sequences, seqs[i]) || c.cells[j].pos > positions[i] {
				mask[i*len+(j-c.curCellRange.min)] = float32(math.Inf(-1))
			}
		}
	}

	return ctx.FromFloatSlice(mask, len, c.curBatchSize)
}

func moveCell(ctx ml.Context, objs []ml.Tensor, src, dst, len int) {
	for _, obj := range objs {
		srcView := obj.View(ctx, int(obj.Stride(2))*src, int(obj.Dim(0)*obj.Dim(1))*len)
		dstView := obj.View(ctx, int(obj.Stride(2))*dst, int(obj.Dim(0)*obj.Dim(1))*len)

		ctx.Forward(srcView.Copy(ctx, dstView))
	}
}

func (c *Causal) defrag() {
	slog.Debug("defragmenting kv cache")

	// Defrag strategy:
	// - Search for empty holes at the beginning of the cache,
	//   filling them with active data starting at the end
	// - If there are contiguous elements that need to be moved,
	//   combine them into a single operation by holding new moves
	//   until we see that the next one is non-contiguous
	// - Fill up the context with the maximum number of operations it
	//   can hold then compute that and continue with a new context
	//
	// We could try to optimize placement by grouping blocks from
	// the same sequences together but most likely the next forward
	// pass will disrupt this anyways, so the real world benefit
	// seems limited as this time.

	ctx := c.model.Backend().NewContext()

	// For every move, 6 tensors are required per layer (2 views and a
	// copy for each of k and v).
	maxMoves := ctx.MaxTensors() / (6 * len(c.keys))
	moves := 0

	var pendingSrc, pendingDst, pendingLen int
	src := len(c.cells) - 1

	for dst := 0; dst < src; dst++ {
		if len(c.cells[dst].sequences) == 0 {
			for ; src > dst; src-- {
				if len(c.cells[src].sequences) != 0 {
					c.cells[dst] = c.cells[src]
					c.cells[src] = cacheCell{}

					if pendingLen > 0 {
						if src == pendingSrc-pendingLen && dst == pendingDst+pendingLen {
							pendingSrc = src
							pendingLen++
							break
						} else {
							moveCell(ctx, c.keys, pendingSrc, pendingDst, pendingLen)
							moveCell(ctx, c.values, pendingSrc, pendingDst, pendingLen)
							moves++
						}
					}

					pendingSrc = src
					pendingDst = dst
					pendingLen = 1

					break
				}
			}
		}

		if moves >= maxMoves {
			ctx.Compute(nil)
			ctx.Close()
			ctx = c.model.Backend().NewContext()

			moves = 0
		}
	}

	if pendingLen > 0 {
		moveCell(ctx, c.keys, pendingSrc, pendingDst, pendingLen)
		moveCell(ctx, c.values, pendingSrc, pendingDst, pendingLen)
		moves++
	}

	if moves > 0 {
		ctx.Compute(nil)
	}
	ctx.Close()

	// Reset range metadata
	for seq := range c.cellRanges {
		seqRange := newRange()

		for i, cell := range c.cells {
			if slices.Contains(cell.sequences, seq) {
				if i < seqRange.min {
					seqRange.min = i
				}
				if i > seqRange.max {
					seqRange.max = i
				}
			}
		}

		c.cellRanges[seq] = seqRange
	}
}

func (c *Causal) SetLayer(layer int) {
	if layer >= len(c.keys) {
		c.keys = append(c.keys, make([]ml.Tensor, layer-len(c.keys)+1)...)
		c.values = append(c.values, make([]ml.Tensor, layer-len(c.values)+1)...)
	}

	c.curLayer = layer
}

func (c *Causal) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
	key := c.keys[c.curLayer]
	value := c.values[c.curLayer]

	key = key.View(ctx, int(key.Stride(2))*c.curCellRange.min,
		int(key.Dim(0)), int(key.Stride(1)),
		int(key.Dim(1)), int(key.Stride(2)),
		int(c.curMask.Dim(0)),
	)

	value = value.View(ctx, int(key.Stride(2))*c.curCellRange.min,
		int(value.Dim(0)), int(value.Stride(1)),
		int(value.Dim(1)), int(value.Stride(2)),
		int(c.curMask.Dim(0)),
	)

	return key, value, c.curMask
}

func (c *Causal) Put(ctx ml.Context, key, value ml.Tensor) {
	if c.curBatchSize != int(key.Dim(2)) {
		panic(fmt.Errorf("inconsistent batch sizes (layer: %v, batch size: %v layer batch size: %v)", c.curLayer, c.curBatchSize, int(key.Dim(2))))
	}

	if c.keys[c.curLayer] == nil || c.values[c.curLayer] == nil {
		c.keys[c.curLayer] = c.cacheCtx.Zeros(c.DType, key.Dim(0), key.Dim(1), int64(c.Capacity))
		c.values[c.curLayer] = c.cacheCtx.Zeros(c.DType, value.Dim(0), value.Dim(1), int64(c.Capacity))
	}

	ctx.Forward(key.Copy(ctx, c.keys[c.curLayer].View(ctx, int(key.Stride(2))*c.curLoc, int(key.Dim(0)*key.Dim(1)*key.Dim(2)))))
	ctx.Forward(value.Copy(ctx, c.values[c.curLayer].View(ctx, int(value.Stride(2))*c.curLoc, int(value.Dim(0)*value.Dim(1)*value.Dim(2)))))
}

func (c *Causal) CopyPrefix(srcSeq, dstSeq int, len int32) {
	seqRange := newRange()

	for i := range c.cells {
		// Remove the contents of dstSeq so that we only have the copied prefix, metadata will be reset at the end
		if slices.Contains(c.cells[i].sequences, dstSeq) {
			c.cells[i].sequences = slices.DeleteFunc(c.cells[i].sequences, func(s int) bool { return s == dstSeq })
		}

		if slices.Contains(c.cells[i].sequences, srcSeq) && c.cells[i].pos < len {
			c.cells[i].sequences = append(c.cells[i].sequences, dstSeq)
			if i < seqRange.min {
				seqRange.min = i
			}
			if i > seqRange.max {
				seqRange.max = i
			}
		}
	}

	c.cellRanges[dstSeq] = seqRange
}

func (c *Causal) shift(seq int, beginIndex, offset int32) error {
	modelShift, ok := c.model.(model.ModelWithShift)
	if !ok {
		return cache.ErrNotSupported
	}

	ctx := c.model.Backend().NewContext()
	defer ctx.Close()

	seqRange := c.cellRanges[seq]
	size := seqRange.max - seqRange.min + 1

	offsets := make([]int32, size)
	for i := range offsets {
		cell := c.cells[seqRange.min+i]

		if slices.Contains(cell.sequences, seq) && cell.pos >= beginIndex {
			offsets[i] = offset
		}
	}

	kShift, err := ctx.FromIntSlice(offsets, len(offsets))
	if err != nil {
		return err
	}

	for i, key := range c.keys {
		if key == nil {
			continue
		}

		key = key.View(ctx, int(key.Stride(2))*seqRange.min,
			int(key.Dim(0)), int(key.Stride(1)),
			int(key.Dim(1)), int(key.Stride(2)),
			size,
		)

		// TODO(jessegross): dequantize once we support data types other than F32 for the cache

		roped, err := modelShift.Shift(ctx, i, key, kShift)
		if err != nil {
			return err
		}

		ctx.Forward(roped.Copy(ctx, key))
	}

	ctx.Compute(nil)

	return nil
}

func (c *Causal) Remove(seq int, beginIndex, endIndex int32) error {
	var offset int32
	if endIndex != math.MaxInt32 {
		offset = beginIndex - endIndex
	}

	seqRange := newRange()

	for i := range c.cells {
		if slices.Contains(c.cells[i].sequences, seq) {
			if c.cells[i].pos >= beginIndex && c.cells[i].pos < endIndex {
				c.cells[i].sequences = slices.DeleteFunc(c.cells[i].sequences, func(s int) bool { return s == seq })
			} else {
				if c.cells[i].pos >= endIndex {
					if slices.ContainsFunc(c.cells[i].sequences, func(s int) bool { return s != seq }) {
						// TODO(jessegross): Need to be careful about data shared between sequences
						panic("shifting on cells shared by multiple sequences not yet implemented")
					}

					c.cells[i].pos += offset
				}
				if i < seqRange.min {
					seqRange.min = i
				}
				if i > seqRange.max {
					seqRange.max = i
				}
			}
		}
	}

	if seqRange == newRange() {
		delete(c.cellRanges, seq)
		return nil
	}

	c.cellRanges[seq] = seqRange

	if endIndex != math.MaxInt32 {
		err := c.shift(seq, endIndex+offset, offset)
		if err != nil {
			return err
		}
	}

	return nil
}
