#pragma once
#include <array>
#include <cstddef>
#include <optional>
#include <mutex>
#include <vector>

// Fixed-capacity ring buffer — O(1) push, O(1) index, thread-safe.
// Oldest entries are silently overwritten once full.

template <typename T, std::size_t Capacity>
class RingBuffer {
public:
    static_assert(Capacity > 0, "Capacity must be positive");

    void push(T item) {
        std::lock_guard<std::mutex> lk(mtx_);
        buf_[head_] = std::move(item);
        head_ = (head_ + 1) % Capacity;
        if (size_ < Capacity) ++size_;
        else tail_ = (tail_ + 1) % Capacity;  // overwrite oldest
    }

    // Returns the i-th element from oldest (0) to newest (size-1).
    std::optional<T> at(std::size_t i) const {
        std::lock_guard<std::mutex> lk(mtx_);
        if (i >= size_) return std::nullopt;
        return buf_[(tail_ + i) % Capacity];
    }

    // Snapshot — returns all elements oldest→newest.
    std::vector<T> snapshot() const {
        std::lock_guard<std::mutex> lk(mtx_);
        std::vector<T> out;
        out.reserve(size_);
        for (std::size_t i = 0; i < size_; ++i)
            out.push_back(buf_[(tail_ + i) % Capacity]);
        return out;
    }

    std::size_t size()     const { std::lock_guard<std::mutex> lk(mtx_); return size_; }
    std::size_t capacity() const { return Capacity; }
    bool        empty()    const { std::lock_guard<std::mutex> lk(mtx_); return size_ == 0; }

    void clear() {
        std::lock_guard<std::mutex> lk(mtx_);
        size_ = head_ = tail_ = 0;
    }

private:
    mutable std::mutex mtx_;
    std::array<T, Capacity> buf_{};
    std::size_t head_ = 0;
    std::size_t tail_ = 0;
    std::size_t size_ = 0;
};