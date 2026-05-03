package brute_force

// #include <stdlib.h>
// #include <cuvs/neighbors/brute_force.h>
import "C"

import (
	"errors"
	"unsafe"

	cuvs "github.com/rapidsai/cuvs/go"
)

// Brute Force KNN Index
type BruteForceIndex struct {
	index   C.cuvsBruteForceIndex_t
	trained bool
}

// Creates a new empty Brute Force KNN Index
func CreateIndex() (*BruteForceIndex, error) {
	var index C.cuvsBruteForceIndex_t

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsBruteForceIndexCreate(&index)))
	if err != nil {
		return nil, err
	}

	return &BruteForceIndex{index: index, trained: false}, nil
}

// Destroys the Brute Force KNN Index
func (index *BruteForceIndex) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsBruteForceIndexDestroy(index.index)))
	if err != nil {
		return err
	}
	return nil
}

// Builds a new Brute Force KNN Index from the dataset for efficient search.
//
// # Arguments
//
// * `Resources` - Resources to use
// * `Dataset` - A row-major matrix on either the host or device to index
// * `metric` - Distance type to use for building the index
// * `metric_arg` - Value of `p` for Minkowski distances - set to 2.0 if not applicable
func BuildIndex[T any](Resources cuvs.Resource, Dataset *cuvs.Tensor[T], metric cuvs.Distance, metric_arg float32, index *BruteForceIndex) error {
	CMetric, exists := cuvs.CDistances[metric]

	if !exists {
		return errors.New("cuvs: invalid distance metric")
	}

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsBruteForceBuild(C.cuvsResources_t(Resources.Resource), (*C.DLManagedTensor)(unsafe.Pointer(Dataset.C_tensor)), C.cuvsDistanceType(CMetric), C.float(metric_arg), index.index)))
	if err != nil {
		return err
	}
	index.trained = true

	return nil
}

// Perform a Nearest Neighbors search on the Index
//
// # Arguments
//
// * `Resources` - Resources to use
// * `queries` - Tensor in device memory to query for
// * `neighbors` - Tensor in device memory that receives the indices of the nearest neighbors
// * `distances` - Tensor in device memory that receives the distances of the nearest neighbors
func SearchIndex[T any](resources cuvs.Resource, index BruteForceIndex, queries *cuvs.Tensor[T], neighbors *cuvs.Tensor[int64], distances *cuvs.Tensor[float32]) error {
	if !index.trained {
		return errors.New("index needs to be built before calling search")
	}

	prefilter := C.cuvsFilter{
		addr:  0,
		_type: C.NO_FILTER,
	}

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsBruteForceSearch(C.ulong(resources.Resource), index.index, (*C.DLManagedTensor)(unsafe.Pointer(queries.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(neighbors.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(distances.C_tensor)), prefilter)))

	return err
}

// Save the index to file.
//
// # Arguments
//
// * `Resources` - Resources to use
// * `filename` - The name of the file to save the index to
// * `index` - The BruteForceIndex to serialize
func Serialize(Resources cuvs.Resource, filename string, index *BruteForceIndex) error {
	if !index.trained {
		return errors.New("index needs to be built before calling serialize")
	}

	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	return cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsBruteForceSerialize(
		C.ulong(Resources.Resource),
		cFilename,
		index.index,
	)))
}

// Load the index from file.
//
// # Arguments
//
// * `Resources` - Resources to use
// * `filename` - The name of the file to load the index from
// * `index` - The BruteForceIndex to load into
func Deserialize(Resources cuvs.Resource, filename string, index *BruteForceIndex) error {
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsBruteForceDeserialize(
		C.ulong(Resources.Resource),
		cFilename,
		index.index,
	)))
	if err != nil {
		return err
	}
	index.trained = true
	return nil
}

// Serialize the index to an in-memory byte slice.
//
// # Arguments
//
// * `Resources` - Resources to use
// * `index` - The BruteForceIndex to serialize
func SerializeToBytes(Resources cuvs.Resource, index *BruteForceIndex) ([]byte, error) {
	if !index.trained {
		return nil, errors.New("index needs to be built before calling serialize")
	}

	var buf *C.uint8_t
	var bufSize C.size_t

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsBruteForceSerializeToBytes(
		C.ulong(Resources.Resource),
		index.index,
		&buf,
		&bufSize,
	)))
	if err != nil {
		return nil, err
	}
	defer C.free(unsafe.Pointer(buf))
	return C.GoBytes(unsafe.Pointer(buf), C.int(bufSize)), nil
}

// Load the index from an in-memory byte slice.
//
// The dtype field of the index must be set before calling this function so that
// the correct internal type is instantiated. The dtype is typically known from
// the context in which the index was originally built and serialized.
//
// # Arguments
//
// * `Resources` - Resources to use
// * `data` - Byte slice containing the serialized index
// * `index` - The BruteForceIndex to load into (dtype must be set beforehand)
func DeserializeFromBytes(Resources cuvs.Resource, data []byte, index *BruteForceIndex) error {
	index.index.dtype.code = C.uchar(1)
	index.index.dtype.bits = C.uchar(8)
	index.index.dtype.lanes = C.ushort(1)

	buf := (*C.uint8_t)(C.CBytes(data))
	defer C.free(unsafe.Pointer(buf))

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsBruteForceDeserializeFromBytes(
		C.ulong(Resources.Resource),
		buf,
		C.size_t(len(data)),
		index.index,
	)))
	if err != nil {
		return err
	}
	index.trained = true
	return nil
}
