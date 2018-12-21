// 2018/12/05 - modified by Tsung-Wei Huang
//   - refactored the code
//   - replaced idler storage with lock-free queue
//   - added load balancing heuristics
//
// 2018/12/03 - created by Tsung-Wei Huang
//   - added WorkStealingQueue class

#pragma once

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <atomic>
#include <memory>
#include <cassert>
#include <deque>
#include <optional>
#include <thread>
#include <algorithm>
#include <set>
#include <numeric>

namespace tf {

// Class: WorkStealingQueue
// Unbounded work stealing queue implementation based on the following paper:
// David Chase and Yossi Lev. Dynamic circular work-stealing deque.
// In SPAA ’05: Proceedings of the seventeenth annual ACM symposium
// on Parallelism in algorithms and architectures, pages 21–28,
// New York, NY, USA, 2005. ACM.
template <typename T>
class WorkStealingQueue {
  public:


    WorkStealingQueue() { }
    WorkStealingQueue(uint64_t) { }

    bool empty() const noexcept { return queue.empty(); }

    int64_t size() const noexcept { return queue.size(); }
    int64_t capacity() const noexcept { return queue.max_size(); }
    
    std::atomic_flag flag = ATOMIC_FLAG_INIT; 

    std::deque<T> queue; 

    template <typename O>
    void push(O&&);

    std::optional<T> pop();
    std::optional<T> steal();
};


template <typename T>
template <typename O>
void WorkStealingQueue<T>::push(O&& o) {
  while (flag.test_and_set(std::memory_order_acquire));
  queue.emplace_back(std::forward<O>(o));
  flag.clear();
}


template <typename T>
std::optional<T> WorkStealingQueue<T>::pop() {
  while (flag.test_and_set(std::memory_order_acquire));
  if(queue.empty()) {
    flag.clear();
    return std::nullopt;
  }
  auto item = queue.back();
  queue.pop_back();
  flag.clear();
  return item;
}


template <typename T>
std::optional<T> WorkStealingQueue<T>::steal() {
  if(queue.empty()) return std::nullopt;
  if(!flag.test_and_set(std::memory_order_acquire)) {
    if(queue.empty()) { 
      flag.clear();
      return std::nullopt;
    }
    auto item = queue.back();
    queue.pop_back();
    flag.clear();
    return item;
  }
  return std::nullopt;
}

// ----------------------------------------------------------------------------


// Class: WorkStealingThreadpool
template <typename Closure>
class WorkStealingThreadpool {

  struct Worker {
    std::condition_variable cv;
    WorkStealingQueue<Closure> queue;
    bool exit  {false};
    bool ready {false};
    uint64_t seed;
    unsigned victim_hint;
  };

  public:

    WorkStealingThreadpool(unsigned);
    ~WorkStealingThreadpool();
    
    size_t num_tasks() const;
    size_t num_workers() const;

    bool is_owner() const;

    template <typename... ArgsT>
    void emplace(ArgsT&&...);

    void batch(std::vector<Closure>&&);

  private:
    
    const std::thread::id _owner  {std::this_thread::get_id()};
    const int load_balancing_factor {4};

    mutable std::mutex _mutex;

    std::vector<Worker> _workers;
    //std::vector<Worker*> _idlers;
    std::vector<std::thread> _threads;

    std::unordered_map<std::thread::id, unsigned> _worker_maps;

    WorkStealingQueue<Worker*> _idlers;
    WorkStealingQueue<Closure> _queue;

    void _spawn(unsigned);
    void _shutdown();
    void _balance_load(unsigned);
    
    unsigned _next_power_of_2(unsigned) const;
    unsigned _randomize(uint64_t&) const;
    unsigned _fast_modulo(uint32_t, uint32_t) const;

    std::optional<Closure> _steal(unsigned);
};

// Constructor
template <typename Closure>
WorkStealingThreadpool<Closure>::WorkStealingThreadpool(unsigned N) : 
  _workers {N},
  _idlers  {_next_power_of_2(std::max(2u, N))} {
  _spawn(N);
}

// Destructor
template <typename Closure>
WorkStealingThreadpool<Closure>::~WorkStealingThreadpool() {
  _shutdown();
}

// Procedure: _shutdown
template <typename Closure>
void WorkStealingThreadpool<Closure>::_shutdown(){

  {
    std::scoped_lock lock(_mutex);
    for(auto& w : _workers){
      w.exit = true;
      w.cv.notify_one();
    }
  } 

  for(auto& t : _threads){
    t.join();
  } 
}

// Function: _randomize
// Generate the random output (using the PCG-XSH-RS scheme)
template <typename Closure>
unsigned WorkStealingThreadpool<Closure>::_randomize(uint64_t& state) const {
  uint64_t current = state;
  state = current * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
  return static_cast<unsigned>((current ^ (current >> 22)) >> (22 + (current >> 61)));
}

// Function: _fast_modulo
// Perfrom fast modulo operation (might be biased but it's ok for our heuristics)
// http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
template <typename Closure>
unsigned WorkStealingThreadpool<Closure>::_fast_modulo(uint32_t x, uint32_t N) const {
  return ((uint64_t) x * (uint64_t) N) >> 32;
}

// Function: _next_power_of_2
template <typename Closure>
unsigned WorkStealingThreadpool<Closure>::_next_power_of_2(unsigned n) const {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
}

// Procedure: _spawn
template <typename Closure>
void WorkStealingThreadpool<Closure>::_spawn(unsigned N) {
  
  // Lock to synchronize all workers before creating _worker_mapss
  std::scoped_lock lock(_mutex);
  
  for(unsigned i=0; i<N; ++i) {
    _threads.emplace_back([this, i, N] () -> void {

      std::optional<Closure> t;
      Worker& w = (_workers[i]);
      w.victim_hint = (i + 1) % N;
      w.seed = i + 1;

      std::unique_lock lock(_mutex, std::defer_lock);

      while(!w.exit) {
        
        // pop from my own queue
        if(t = w.queue.pop(); !t) {
          // steal from others
          t = _steal(i);
        }
        
        // no tasks
        if(!t) {
          lock.lock();
          if(_queue.empty()) {
            w.ready = false;
            //_idlers.push_back(&w);
            _idlers.push(&w);
            while(!w.ready && !w.exit) {
              w.cv.wait(lock);
            }
          }
          lock.unlock();
        }

        while(t) {
          (*t)();
          t = w.queue.pop();
        }
      } // End of while ------------------------------------------------------ 

    });     

    _worker_maps.insert({_threads.back().get_id(), i});
  }
}

// Function: is_owner
template <typename Closure>
bool WorkStealingThreadpool<Closure>::is_owner() const {
  return std::this_thread::get_id() == _owner;
}

// Function: num_workers
template <typename Closure>
size_t WorkStealingThreadpool<Closure>::num_workers() const { 
  return _threads.size();  
}

// Procedure: balance_load
template <typename Closure>
void WorkStealingThreadpool<Closure>::_balance_load(unsigned me) {

  int factor = load_balancing_factor;
  
  while(_workers[me].queue.size() > factor) {
    if(auto idler = _idlers.steal(); idler) {
      (*idler)->ready = true;
      (*idler)->victim_hint = me;
      (*idler)->cv.notify_one();
      factor += load_balancing_factor;
    }
    else break;
  }
}
  
// Function: _steal
template <typename Closure>
std::optional<Closure> WorkStealingThreadpool<Closure>::_steal(unsigned thief) {

  std::optional<Closure> task;
  
  for(int round=0; round<1024; ++round) {

    // try getting a task from the centralized queue
    if(task = _queue.steal(); task) {
      return task;
    }

    // try stealing a task from other workers
    unsigned victim = _workers[thief].victim_hint;

    for(unsigned i=0; i<_workers.size(); i++){

      if(victim != thief) {
        if(task = _workers[victim].queue.steal(); task){
          _workers[thief].victim_hint = victim;
          return task;
        }
      }

      victim += 1;
      if(victim == _workers.size()){
        victim = 0;
      }
    }

    // nothing happens this round
    std::this_thread::yield();
  }
  
  return std::nullopt; 
}

// Procedure: emplace
template <typename Closure>
template <typename... ArgsT>
void WorkStealingThreadpool<Closure>::emplace(ArgsT&&... args){

  //no worker thread available
  if(num_workers() == 0){
    Closure{std::forward<ArgsT>(args)...}();
    return;
  }

  // caller is not the owner
  if(auto tid = std::this_thread::get_id(); tid != _owner){

    // the caller is the worker of the threadpool
    if(auto itr = _worker_maps.find(tid); itr != _worker_maps.end()){

      unsigned me = itr->second;

      _workers[me].queue.push(Closure{std::forward<ArgsT>(args)...});
      
      // load balancing
      _balance_load(me);
      
      return;
    }
  }

  if(auto idler = _idlers.steal(); idler) {
    (*idler)->ready = true;
    (*idler)->queue.push(Closure{std::forward<ArgsT>(args)...});
    (*idler)->cv.notify_one(); 
  }
  else {
    std::scoped_lock lock(_mutex);
    _queue.push(Closure{std::forward<ArgsT>(args)...});
  }
}

// Procedure: batch
template <typename Closure>
void WorkStealingThreadpool<Closure>::batch(std::vector<Closure>&& tasks) {

  //no worker thread available
  if(num_workers() == 0){
    for(auto &t: tasks){
      t();
    }
    return;
  }
  
  // caller is not the owner
  if(auto tid = std::this_thread::get_id(); tid != _owner){

    // the caller is the worker of the threadpool
    if(auto itr = _worker_maps.find(tid); itr != _worker_maps.end()){

      unsigned me = itr->second;

      for(auto& t : tasks) {
        _workers[me].queue.push(std::move(t));
      }
      
      // load balancing 
      _balance_load(me);

      return;
    }
  }
  
  {
    std::scoped_lock lock(_mutex);
    
    for(auto& task : tasks) {
      _queue.push(std::move(task));
    }
  }

  while(!_queue.empty()) {
    if(auto idler = _idlers.steal(); idler) {
      (*idler)->ready = true;
      (*idler)->cv.notify_one();
    }
    else {
      break;
    }
  }
} 

};  // end of namespace tf. ---------------------------------------------------





