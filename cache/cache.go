package cache

import (
	"errors"

	"github.com/ollama/ollama/ml"
)

var (
	ErrKvCacheFull  = errors.New("could not find a kv cache slot")
	ErrNotSupported = errors.New("model does not support operation")
)

type Cache interface {
	// ** used by model implementations **

	// Sets the active layer of the cache
	SetLayer(layer int)

	// Returns the history of key and value tensors plus a mask
	//
	// The tensors are of shape embed dim, kv heads, batch size
	// The mask is of shape history size, batch size
	Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor)

	// Stores a batch of key and value in the cache
	//
	// The tensors must be of shape embed dim, kv heads, batch size
	Put(ctx ml.Context, key, value ml.Tensor)

	// ** cache management **

	// Closes the cache and frees resources associated with it
	Close()

	// Called before the start of the model's forward pass. For each
	// token in the coming batch, there must be a corresponding entry
	// in positions and seqs.
	StartForward(ctx ml.Context, positions []int32, seqs []int) error

	// Copies tokens in the range [0, len) from srcSeq to dstSeq
	CopyPrefix(srcSeq, dstSeq int, len int32)

	// Removes tokens in the range [beginIndex, endIndex) from seq. Set
	// endIndex to math.MaxInt32 to remove everything starting at beginIndex.
	//
	// If an error occurs, the entire context for the sequence should be
	// removed by calling Remove(seq, 0, math.MaxInt32)
	Remove(seq int, beginIndex, endIndex int32) error
}
