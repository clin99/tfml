// 2018/12/04 - modified by Tsung-Wei Huang
//   - replaced privatized threadpool with work stealing threadpool
//
// 2018/12/03 - modified by Tsung-Wei Huang
//   - added work stealing queue tests
//
// 2018/11/29 - modified by Chun-Xun Lin
//   - added batch tests
//
// 2018/10/04 - modified by Tsung-Wei Huang
//   - removed binary tree tests
//   - removed spawn/shutdown tests
//   - removed siltne_async and async tests
//   - added emplace test
//   - adopted the new thread pool implementation
//
// 2018/09/29 - modified by Tsung-Wei Huang
//   - added binary tree tests
//   - added worker queue tests
//   - added external thread tests
//   - refactored threadpool tests
// 
// 2018/09/13 - modified by Tsung-Wei Huang & Chun-Xun
//   - added tests for ownership
//   - modified spawn-shutdown tests
//
// 2018/09/10 - modified by Tsung-Wei Huang
//   - added tests for SpeculativeThreadpool
//   - added dynamic tasking tests
//   - added spawn and shutdown tests
//
// 2018/09/02 - created by Guannan
//   - test_silent_async
//   - test_async
//   - test_wait_for_all

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/threadpool/threadpool.hpp>

// ============================================================================
// WorkStealingQueue tests
// ============================================================================

// Procedure: wsq_test_owner
void wsq_test_owner() {

  int64_t cap = 2;
  
  tf::WorkStealingQueue<int> queue(cap);
  std::deque<int> gold;

  REQUIRE(queue.capacity() == 2);
  REQUIRE(queue.empty());

  for(int i=2; i<=(1<<20); i <<= 1) {

    REQUIRE(queue.empty());

    for(int j=0; j<i; ++j) {
      queue.push(j);
    }

    for(int j=0; j<i; ++j) {
      auto item = queue.pop();
      REQUIRE((item && *item == i-j-1));
    }
    REQUIRE(!queue.pop());
    
    REQUIRE(queue.empty());
    for(int j=0; j<i; ++j) {
      queue.push(j);
    }
    
    for(int j=0; j<i; ++j) {
      auto item = queue.steal();
      REQUIRE((item && *item == j));
    }
    REQUIRE(!queue.pop());

    REQUIRE(queue.empty());

    for(int j=0; j<i; ++j) {
      // enqueue 
      if(auto dice = ::rand()%3; dice == 0) {
        queue.push(j);
        gold.push_back(j);
      }
      // pop back
      else if(dice == 1) {
        auto item = queue.pop();
        if(gold.empty()) {
          REQUIRE(!item);
        }
        else {
          REQUIRE(*item == gold.back());
          gold.pop_back();
        }
      }
      // pop front
      else {
        auto item = queue.steal();
        if(gold.empty()) {
          REQUIRE(!item);
        }
        else {
          REQUIRE(*item == gold.front());
          gold.pop_front();
        }
      }

      REQUIRE(queue.size() == (int)gold.size());
    }

    while(!queue.empty()) {
      auto item = queue.pop();
      REQUIRE((item && *item == gold.back()));
      gold.pop_back();
    }

    REQUIRE(gold.empty());
    
    REQUIRE(queue.capacity() == i);
  }
}

// Procedure: wsq_test_n_thieves
void wsq_test_n_thieves(int N) {

  int64_t cap = 2;
  
  tf::WorkStealingQueue<int> queue(cap);

  REQUIRE(queue.capacity() == 2);
  REQUIRE(queue.empty());

  for(int i=2; i<=(1<<20); i <<= 1) {

    REQUIRE(queue.empty());

    int p = 0;

    std::vector<std::deque<int>> cdeqs(N);
    std::vector<std::thread> consumers;
    std::deque<int> pdeq;

    auto num_stolen = [&] () {
      int total = 0;
      for(const auto& cdeq : cdeqs) {
        total += cdeq.size();
      }
      return total;
    };
    
    for(int n=0; n<N; n++) {
      consumers.emplace_back([&, n] () {
        while(num_stolen() + (int)pdeq.size() != i) {
          if(auto dice = ::rand() % 4; dice == 0) {
            if(auto item = queue.steal(); item) {
              cdeqs[n].push_back(*item);
            }
          }
        }
      });
    }

    std::thread producer([&] () {
      while(p < i) { 
        if(auto dice = ::rand() % 4; dice == 0) {
          queue.push(p++); 
        }
        else if(dice == 1) {
          if(auto item = queue.pop(); item) {
            pdeq.push_back(*item);
          }
        }
      }
    });

    producer.join();

    for(auto& c : consumers) {
      c.join();
    }

    REQUIRE(queue.empty());
    REQUIRE(queue.capacity() <= i);

    std::set<int> set;
    
    for(const auto& cdeq : cdeqs) {
      for(auto k : cdeq) {
        set.insert(k);
      }
    }
    
    for(auto k : pdeq) {
      set.insert(k);
    }

    for(int j=0; j<i; ++j) {
      REQUIRE(set.find(j) != set.end());
    }

    REQUIRE((int)set.size() == i);
  }
}

// ----------------------------------------------------------------------------
// Testcase: WSQTest.Owner
// ----------------------------------------------------------------------------
TEST_CASE("WSQ.Owner" * doctest::timeout(300)) {
  wsq_test_owner();
}

// ----------------------------------------------------------------------------
// Testcase: WSQTest.1Thief
// ----------------------------------------------------------------------------
TEST_CASE("WSQ.1Thief" * doctest::timeout(300)) {
  wsq_test_n_thieves(1);
}

// ----------------------------------------------------------------------------
// Testcase: WSQTest.2Thieves
// ----------------------------------------------------------------------------
TEST_CASE("WSQ.2Thieves" * doctest::timeout(300)) {
  wsq_test_n_thieves(2);
}

// ----------------------------------------------------------------------------
// Testcase: WSQTest.3Thieves
// ----------------------------------------------------------------------------
TEST_CASE("WSQ.3Thieves" * doctest::timeout(300)) {
  wsq_test_n_thieves(3);
}

// ----------------------------------------------------------------------------
// Testcase: WSQTest.4Thieves
// ----------------------------------------------------------------------------
TEST_CASE("WSQ.4Thieves" * doctest::timeout(300)) {
  wsq_test_n_thieves(4);
}


// ============================================================================
// Threadpool tests
// ============================================================================

// Procedure: test_ownership
template <typename ThreadpoolType>
void test_ownership(ThreadpoolType& tp) {
  
  REQUIRE(tp.is_owner()); 
  
  tp.emplace([&](){
    if(tp.num_workers() == 0) {
      REQUIRE(tp.is_owner());
    }
    else {
      REQUIRE(!tp.is_owner());
    }
  });

  std::vector<std::thread> threads;
  for(int i=0; i<10; ++i) {
    threads.emplace_back([&] () {
      REQUIRE(!tp.is_owner());
    });
  }
  for(auto& t : threads) {
    t.join();
  }
}

// Procedure: test_emplace
template <typename ThreadpoolType>
void test_emplace(ThreadpoolType& tp) {

  constexpr size_t num_tasks = 1024;

  std::atomic<size_t> counter{0};
  
  for(size_t i=0; i<num_tasks; i++){
    tp.emplace([&counter](){ 
      counter.fetch_add(1, std::memory_order_relaxed); }
    );
  }

  while(counter != num_tasks);
}

// Procedure: test_dynamic_tasking
template <typename T>
void test_dynamic_tasking(T& threadpool) {

  std::atomic<size_t> sum {0};
  std::atomic<size_t> cnt {0};

  std::function<void(int)> insert;
  std::promise<int> promise;
  auto future = promise.get_future();
  
  insert = [&threadpool, &insert, &sum, &promise, &cnt] (int i) {
    if(i > 0) {
      ++cnt;
      threadpool.emplace([i=i-1, &insert] () {
        insert(i);
      });
    }
    else {
      if(auto s = ++sum; s == threadpool.num_workers()) {
        promise.set_value(1);
      }
    }
  };

  if(auto W = threadpool.num_workers(); W > 0) {
    for(size_t i=0; i<threadpool.num_workers(); i++){
      insert(100);
    }
  }
  else {
    promise.set_value(1);
  }

  // synchronize until all tasks finish
  REQUIRE(future.get() == 1);
  REQUIRE(cnt == 100 * threadpool.num_workers());
  REQUIRE(sum == threadpool.num_workers());
}

// Procedure: test_external_threads
template <typename T>
void test_external_threads(T& threadpool) {

  constexpr int num_tasks = 65536;

  std::vector<std::thread> threads;
  std::atomic<size_t> sum {0};

  for(int i=0; i<10; ++i) {
    threads.emplace_back([&] () {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      for(int j=0; j<num_tasks; ++j) {
        threadpool.emplace([&] () { 
          sum.fetch_add(1, std::memory_order_relaxed);
        });
      }
    });
  }
  
  // master thread to insert
  for(int j=0; j<num_tasks; ++j) {
    threadpool.emplace([&] () {
      sum.fetch_add(1, std::memory_order_relaxed);
    });
  }

  // worker thread to insert
  for(int i=0; i<10; ++i) {
    threadpool.emplace([&] () {
      for(int j=0; j<num_tasks; ++j) {
        threadpool.emplace([&] () {
          sum.fetch_add(1, std::memory_order_relaxed);
        });
      }
    });
  }
  
  for(auto& t : threads) {
    t.join();
  }

  while(sum != num_tasks * 10 * 2 + num_tasks) ;
}


// Procedure: test_external_threads
template <typename T>
void test_batch_insertion(T& threadpool) {
  constexpr int num_iterations = 50;

  size_t total {0};
  std::atomic<size_t> count {0};
  for(size_t i=1; i<num_iterations; i++) {

    std::vector<std::function<void()>> funs;
    for(size_t j=0; j<i; j++) {
      funs.emplace_back([&](){count++;});
    }

    threadpool.batch(std::move(funs));
    total += i;
  }

  while(count != total);
}
  
// Procedure: test_threadpool
template <typename T>
void test_threadpool() {  

  SUBCASE("Ownership") {
    printf("Ownership\n");
    for(unsigned i=0; i<=4; ++i) {
      T tp(i);
      test_ownership(tp);
    }
  }

  SUBCASE("Emplace") {
    printf("Emplace\n");
    for(unsigned i=0; i<=4; ++i) {
      T tp(i);
      test_emplace(tp);
    }
  }


  SUBCASE("DynamicTasking") {
    printf("DynamicTasking\n");
    for(unsigned i=0; i<=4; ++i) {
      T tp(i);
      test_dynamic_tasking(tp);
    }
  }

  SUBCASE("ExternalThreads") {
    printf("ExternalThreads\n");
    for(unsigned i=0; i<=4; ++i) {
      T tp(i);
      test_external_threads(tp);
    }
  }

  SUBCASE("Batch") {
    printf("Batch\n");
    for(unsigned i=0; i<=4; ++i) {
      T tp(i);
      test_batch_insertion(tp);
    }
  }
}

// ----------------------------------------------------------------------------
// Testcase: SimpleThreadpool
// ----------------------------------------------------------------------------
TEST_CASE("SimpleThreadpool" * doctest::timeout(300)) {
  test_threadpool<tf::SimpleThreadpool<std::function<void()>>>();
}

// ----------------------------------------------------------------------------
// Testcase: ProactiveThreadpool
// ----------------------------------------------------------------------------
TEST_CASE("ProactiveThreadpool" * doctest::timeout(300)) {
  test_threadpool<tf::ProactiveThreadpool<std::function<void()>>>();
}

// ----------------------------------------------------------------------------
// Testcase: SpeculativeThreadpool
// ----------------------------------------------------------------------------
TEST_CASE("SpeculativeThreadpool" * doctest::timeout(300)) {
  test_threadpool<tf::SpeculativeThreadpool<std::function<void()>>>();
}

// ----------------------------------------------------------------------------
// Testcase: WorkStealingThreadpool
// ----------------------------------------------------------------------------
TEST_CASE("WorkStealingThreadpool" * doctest::timeout(300)) {
  test_threadpool<tf::WorkStealingThreadpool<std::function<void()>>>();
}





