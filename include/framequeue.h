#ifndef FRAMEQUEUE_H
#define FRAMEQUEUE_H

#include <queue>
#include <mutex>

#include <opencv2/core.hpp>

template <typename T>
class FrameQueue : private std::queue<T>
{
public:
    // constructor / destructor
    FrameQueue();
    ~FrameQueue();
    // methods 
    void push(const T &item);
    T get();
    bool isEmpty();
    //void clear();

private:
    std::mutex _mutex;
    //std::queue<T> _queue;
};

// ---- Template class function definitions

// constructor
template <typename T>
FrameQueue<T>::FrameQueue(){}

// destructor
template <typename T>
FrameQueue<T>::~FrameQueue(){}

template <typename T>
void FrameQueue<T>::push(const T &item)
{
    // perform queue modification under the lock
    std::lock_guard<std::mutex> lckg(_mutex);
    std::queue<T>::push(item);
}

template <typename T>
T FrameQueue<T>::get()
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
    return this->empty();
}

#endif