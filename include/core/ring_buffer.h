#pragma once
#include <array>
#include <cstddef>
#include <mutex>

// Thread-safe fixed-capacity ring buffer.
// When full, oldest entry is silently overwritten.
template<typename T, size_t Capacity>
class RingBuffer {
    static_assert(Capacity > 0, "Capacity must be positive");
public:
    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        data_[head_] = std::move(item);
        head_ = (head_ + 1) % Capacity;
        if (size_ < Capacity) {
            ++size_;
        } else {
            tail_ = (tail_ + 1) % Capacity;
        }
    }

    // Iterate oldest→newest under lock.
    template<typename Fn>
    void for_each(Fn&& fn) const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < size_; ++i) {
            fn(data_[(tail_ + i) % Capacity]);
        }
    }

    // Return copy of element at logical index (0 = oldest).
    T at(size_t idx) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_[(tail_ + idx) % Capacity];
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_ == 0;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        head_ = tail_ = size_ = 0;
    }

    static constexpr size_t capacity() { return Capacity; }

private:
    std::array<T, Capacity> data_{};
    size_t head_ = 0;
    size_t tail_ = 0;
    size_t size_ = 0;
    mutable std::mutex mutex_;
};
