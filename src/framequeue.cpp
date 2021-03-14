#include "framequeue.h"

#include <chrono>
#include <iostream>

// constructor / destructor
template <typename T>
FrameQueue<T>::FrameQueue(){}

template <typename T>
FrameQueue<T>::~FrameQueue(){}

// methods
template <typename T>
void FrameQueue<T>::pushFrame(const T &item)
{
    // perform queue modification under the lock
    std::lock_guard<std::mutex> lckg(_mutex);
    std::queue<T>::push(item); 
}

template <typename T>
T FrameQueue<T>::getFrame()
{
    // perform queue modification under the lock
    std::lock_guard<std::mutex> lckg(_mutex);
    // first in first out
    T item = std::move(this->front());
    this->pop();
    return item;
}

template <typename T>
bool FrameQueue<T>::isEmpty()
{
    return this->isEmpty();
}