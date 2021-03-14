#ifndef FRAMEQUEUE_H
#define FRAMEQUEUE_H

#include <queue>
#include <mutex>

#include <opencv2/core.hpp>

template <class T>
class FrameQueue : private std::queue<T>
{
public:
    // constructor / destructor
    FrameQueue();
    ~FrameQueue();

    // methods 
    void pushFrame(const T &item);
    T getFrame();
    bool isEmpty();

private:
    std::mutex _mutex;
};

#endif