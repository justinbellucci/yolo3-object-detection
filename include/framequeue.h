#ifndef FRAMEQUEUE_H
#define FRAMEQUEUE_H

#include <queue>
#include <mutex>

#include <opencv2/core.hpp>

template <class T>
class FrameQueue 
{
public:
    // constructor / destructor
    FrameQueue();
    ~FrameQueue();
    // // copy constructor
    // FrameQueue(const FrameQueue &source) = delete;
    // // copy assignment operator
    // FrameQueue &operator=(const FrameQueue &source) = delete;
    // // move constructor
    // FrameQueue(FrameQueue &&source) = delete;
    // // move assignment operator
    // FrameQueue &operator=(FrameQueue &&source) = delete;

    // methods 
    void pushFrame(const T &item);
    T getFrame();
    bool isEmpty();

private:
    std::mutex _mutex;
    std::queue<T> _queue;
};

// ---- Template class function definitions

// constructor
template <typename T>
FrameQueue<T>::FrameQueue(){}

// destructor
template <typename T>
FrameQueue<T>::~FrameQueue(){}

template <typename T>
void FrameQueue<T>::pushFrame(const T &item)
{
    // perform queue modification under the lock
    std::lock_guard<std::mutex> lckg(_mutex);
    _queue.push(item);
}

template <typename T>
T FrameQueue<T>::getFrame()
{
    // perform queue modification under the lock
    std::lock_guard<std::mutex> lckg(_mutex);
    // first in first out
    T item = std::move(_queue.front());
    _queue.pop();

    return item;
}

#endif