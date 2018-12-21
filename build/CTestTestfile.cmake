# CMake generated Testfile for 
# Source directory: /Users/clin99/cpp-taskflow
# Build directory: /Users/clin99/cpp-taskflow/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(builder "/Users/clin99/cpp-taskflow/unittest/taskflow" "-tc=Builder")
add_test(dispatch "/Users/clin99/cpp-taskflow/unittest/taskflow" "-tc=Dispatch")
add_test(executor "/Users/clin99/cpp-taskflow/unittest/taskflow" "-tc=Executor")
add_test(parallel_for "/Users/clin99/cpp-taskflow/unittest/taskflow" "-tc=ParallelFor")
add_test(reduce "/Users/clin99/cpp-taskflow/unittest/taskflow" "-tc=Reduce")
add_test(reduce_min "/Users/clin99/cpp-taskflow/unittest/taskflow" "-tc=ReduceMin")
add_test(reduce_max "/Users/clin99/cpp-taskflow/unittest/taskflow" "-tc=ReduceMax")
add_test(joined_subflow "/Users/clin99/cpp-taskflow/unittest/taskflow" "-tc=JoinedSubflow")
add_test(detached_subflow "/Users/clin99/cpp-taskflow/unittest/taskflow" "-tc=DetachedSubflow")
add_test(WorkStealingQueue.Owner "/Users/clin99/cpp-taskflow/unittest/threadpool" "-tc=WSQ.Owner")
add_test(WorkStealingQueue.1Thief "/Users/clin99/cpp-taskflow/unittest/threadpool" "-tc=WSQ.1Thief")
add_test(WorkStealingQueue.2Thieves "/Users/clin99/cpp-taskflow/unittest/threadpool" "-tc=WSQ.2Thieves")
add_test(WorkStealingQueue.3Thieves "/Users/clin99/cpp-taskflow/unittest/threadpool" "-tc=WSQ.3Thieves")
add_test(WorkStealingQueue.4Thieves "/Users/clin99/cpp-taskflow/unittest/threadpool" "-tc=WSQ.4Thieves")
add_test(simple_threadpool "/Users/clin99/cpp-taskflow/unittest/threadpool" "-tc=SimpleThreadpool")
add_test(proactive_threadpool "/Users/clin99/cpp-taskflow/unittest/threadpool" "-tc=ProactiveThreadpool")
add_test(speculative_threadpool "/Users/clin99/cpp-taskflow/unittest/threadpool" "-tc=SpeculativeThreadpool")
add_test(work_stealing_threadpool "/Users/clin99/cpp-taskflow/unittest/threadpool" "-tc=WorkStealingThreadpool")
add_test(threadpool_cxx14_basic "/Users/clin99/cpp-taskflow/unittest/threadpool_cxx14" "-tc=Threadpool.Basic")
add_test(threadpool_cxx14_wait_for_all "/Users/clin99/cpp-taskflow/unittest/threadpool_cxx14" "-tc=Threadpool.WaitForAll")
