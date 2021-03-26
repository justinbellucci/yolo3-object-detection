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
    void clear();
    float getFPS();

private:
    std::mutex _mutex;
    cv::TickMeter _tickMeter;
    unsigned int _counter;
};

// ---- Template class function definitions

// constructor
template <typename T>
FrameQueue<T>::FrameQueue() : _counter(0) {}

// destructor
template <typename T>
FrameQueue<T>::~FrameQueue(){}

template <typename T>
void FrameQueue<T>::push(const T &item)
{
    // perform queue modification under the lock
    std::lock_guard<std::mutex> lckg(_mutex);
    std::queue<T>::push(item);
    _counter += 1;
    if(_counter == 1)
    {
        _tickMeter.reset();
        _tickMeter.start();
    }
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

template <typename T>
void FrameQueue<T>::clear()
{
    std::lock_guard<std::mutex> lckg(_mutex);
    while (!this->empty())
    {
        this->pop();
    }
}

template <typename T>
float FrameQueue<T>::getFPS()
{
    _tickMeter.stop();
    double fps = _counter / _tickMeter.getTimeSec();
    _tickMeter.start();
    return static_cast<float>(fps);
}

#endif