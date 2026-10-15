//
//  Tensors.swift
//  StableAudioKitCoreAI
//
//  Moving numbers in and out of Core AI's NDArray: host arrays to tensors,
//  tensors back to host arrays (any strides), and zero tensors.
//

import CoreAI
import Foundation

@available(macOS 27.0, iOS 27.0, *)
enum Tensors {

    /// A float32 tensor holding `values` in row-major order.
    static func floats(_ values: [Float], shape: [Int]) -> NDArray {
        precondition(values.count == shape.reduce(1, *), "\(values.count) values for shape \(shape)")
        var array = NDArray(shape: shape, scalarType: .float32)
        let view = array.mutableView(as: Float.self)
        view.withUnsafeMutablePointer { pointer, _, strides in
            values.withUnsafeBufferPointer { source in
                copy(from: source.baseAddress!, to: pointer, shape: shape, strides: list(strides))
            }
        }
        return array
    }

    /// A float32 tensor from float64 host values.
    static func floats(_ values: [Double], shape: [Int]) -> NDArray {
        floats(values.map(Float.init), shape: shape)
    }

    /// An int32 tensor holding `values` in row-major order.
    static func ints(_ values: [Int32], shape: [Int]) -> NDArray {
        precondition(values.count == shape.reduce(1, *), "\(values.count) values for shape \(shape)")
        var array = NDArray(shape: shape, scalarType: .int32)
        let view = array.mutableView(as: Int32.self)
        view.withUnsafeMutablePointer { pointer, _, strides in
            values.withUnsafeBufferPointer { source in
                copy(from: source.baseAddress!, to: pointer, shape: shape, strides: list(strides))
            }
        }
        return array
    }

    /// A float32 tensor of `other`'s shape holding zeros.
    static func zeros(like other: NDArray) throws -> NDArray {
        guard other.scalarType == .float32 else { throw TensorError.scalarType("\(other.scalarType)") }
        return floats([Float](repeating: 0, count: other.shape.reduce(1, *)), shape: other.shape)
    }

    /// A float32 tensor's values in row-major order, whatever its strides.
    static func values(_ array: NDArray) throws -> [Float] {
        guard array.scalarType == .float32 else {
            throw TensorError.scalarType("\(array.scalarType)")
        }
        let shape = array.shape
        let count = shape.reduce(1, *)
        var out = [Float](repeating: 0, count: count)
        let view = array.view(as: Float.self)
        view.withUnsafePointer { pointer, _, strides in
            out.withUnsafeMutableBufferPointer { target in
                gather(from: pointer, strides: list(strides), shape: shape, into: target.baseAddress!)
            }
        }
        return out
    }

    enum TensorError: Error, CustomStringConvertible {
        case scalarType(String)
        var description: String {
            switch self { case .scalarType(let type): "expected a float32 output, got \(type)" }
        }
    }

    // MARK: Strided copies

    /// A span's values.
    private static func list(_ span: Span<Int>) -> [Int] {
        (0..<span.count).map { span[$0] }
    }

    /// Row-major `source` into a strided `target`.
    private static func copy<T>(from source: UnsafePointer<T>, to target: UnsafeMutablePointer<T>, shape: [Int], strides: [Int]) {
        walk(shape: shape, strides: strides) { flat, offset in target[offset] = source[flat] }
    }

    /// A strided `source` into row-major `target`.
    private static func gather<T>(from source: UnsafePointer<T>, strides: [Int], shape: [Int], into target: UnsafeMutablePointer<T>) {
        walk(shape: shape, strides: strides) { flat, offset in target[flat] = source[offset] }
    }

    /// Calls `body` with each element's row-major index and its strided
    /// offset. Contiguous tensors take one pass.
    private static func walk(shape: [Int], strides: [Int], _ body: (Int, Int) -> Void) {
        let count = shape.reduce(1, *)
        guard count > 0 else { return }
        var contiguous = [Int](repeating: 1, count: shape.count)
        for axis in stride(from: shape.count - 2, through: 0, by: -1) {
            contiguous[axis] = contiguous[axis + 1] * shape[axis + 1]
        }
        let isContiguous = strides.isEmpty || zip(shape, zip(strides, contiguous)).allSatisfy { $0.0 == 1 || $0.1.0 == $0.1.1 }
        if isContiguous {
            for index in 0..<count { body(index, index) }
            return
        }
        var position = [Int](repeating: 0, count: shape.count)
        for index in 0..<count {
            var offset = 0
            for axis in position.indices { offset += position[axis] * strides[axis] }
            body(index, offset)
            var axis = shape.count - 1
            while axis >= 0 {
                position[axis] += 1
                if position[axis] < shape[axis] { break }
                position[axis] = 0
                axis -= 1
            }
        }
    }
}
