# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for PriorityRequestQueue lazy deletion optimization."""

import random
import time
from typing import List

import pytest

from vllm.v1.core.sched.request_queue import PriorityRequestQueue
from vllm.v1.request import Request, RequestStatus


def create_mock_request(request_id: str, priority: int = 1,
                        arrival_time: float = None) -> Request:
    """Create a mock request for testing."""
    if arrival_time is None:
        arrival_time = time.time()
    
    # Create a minimal Request object for testing
    request = Request.__new__(Request)
    request.request_id = request_id
    request.priority = priority
    request.arrival_time = arrival_time
    request.status = RequestStatus.WAITING
    return request


class TestPriorityQueueLazyDeletion:
    """Test lazy deletion functionality in PriorityRequestQueue."""

    def test_basic_operations(self):
        """Test basic queue operations work correctly."""
        queue = PriorityRequestQueue()
        
        # Test empty queue
        assert len(queue) == 0
        assert not queue
        with pytest.raises(IndexError):
            queue.pop_request()
        with pytest.raises(IndexError):
            queue.peek_request()
        
        # Add requests
        req1 = create_mock_request("req1", priority=2, arrival_time=1.0)
        req2 = create_mock_request("req2", priority=1, arrival_time=2.0)
        req3 = create_mock_request("req3", priority=3, arrival_time=3.0)
        
        queue.add_request(req1)
        queue.add_request(req2)
        queue.add_request(req3)
        
        assert len(queue) == 3
        assert bool(queue)
        
        # Test priority ordering (lower priority value = higher priority)
        assert queue.peek_request() == req2  # priority 1
        assert queue.pop_request() == req2
        assert queue.pop_request() == req1  # priority 2
        assert queue.pop_request() == req3  # priority 3
        
        assert len(queue) == 0
        assert not queue

    def test_remove_single_request(self):
        """Test removing a single request using lazy deletion."""
        queue = PriorityRequestQueue()
        
        requests = [
            create_mock_request(f"req{i}", priority=i, arrival_time=float(i))
            for i in range(5)
        ]
        
        for req in requests:
            queue.add_request(req)
        
        # Remove middle request
        queue.remove_request(requests[2])
        assert len(queue) == 4
        
        # Verify it's not returned when popping
        popped = []
        while queue:
            popped.append(queue.pop_request())
        
        assert requests[2] not in popped
        assert len(popped) == 4

    def test_remove_multiple_requests(self):
        """Test removing multiple requests in batch."""
        queue = PriorityRequestQueue()
        
        requests = [
            create_mock_request(f"req{i}", priority=i % 3, arrival_time=float(i))
            for i in range(10)
        ]
        
        for req in requests:
            queue.add_request(req)
        
        # Remove half of the requests
        to_remove = requests[::2]  # Every other request
        queue.remove_requests(to_remove)
        
        assert len(queue) == 5
        
        # Verify removed requests are not returned
        remaining = []
        while queue:
            remaining.append(queue.pop_request())
        
        for req in to_remove:
            assert req not in remaining
        assert len(remaining) == 5

    def test_rebuild_threshold(self):
        """Test that heap rebuilds when threshold is exceeded."""
        queue = PriorityRequestQueue(rebuild_threshold=0.5)
        
        # Add 100 requests
        requests = [
            create_mock_request(f"req{i}", priority=i % 10, arrival_time=float(i))
            for i in range(100)
        ]
        
        for req in requests:
            queue.add_request(req)
        
        initial_heap_size = len(queue._heap)
        
        # Remove 60 requests (60% > 50% threshold)
        to_remove = requests[:60]
        queue.remove_requests(to_remove)
        
        # After rebuild, heap should be smaller
        assert len(queue._heap) < initial_heap_size
        assert len(queue._deleted) == 0  # Deleted set should be cleared
        assert len(queue) == 40  # Correct count

    def test_peek_with_deleted_items(self):
        """Test peek correctly skips deleted items."""
        queue = PriorityRequestQueue()
        
        req1 = create_mock_request("req1", priority=1, arrival_time=1.0)
        req2 = create_mock_request("req2", priority=2, arrival_time=2.0)
        req3 = create_mock_request("req3", priority=3, arrival_time=3.0)
        
        queue.add_request(req1)
        queue.add_request(req2)
        queue.add_request(req3)
        
        # Remove the highest priority item
        queue.remove_request(req1)
        
        # Peek should return the next valid item
        assert queue.peek_request() == req2

    def test_iteration_skips_deleted(self):
        """Test iteration correctly skips deleted items."""
        queue = PriorityRequestQueue()
        
        requests = [
            create_mock_request(f"req{i}", priority=i, arrival_time=float(i))
            for i in range(10)
        ]
        
        for req in requests:
            queue.add_request(req)
        
        # Remove some requests
        to_remove = [requests[2], requests[5], requests[7]]
        queue.remove_requests(to_remove)
        
        # Iterate and verify deleted items are skipped
        iterated = list(queue)
        assert len(iterated) == 7
        for req in to_remove:
            assert req not in iterated

    def test_re_add_deleted_request(self):
        """Test re-adding a previously deleted request."""
        queue = PriorityRequestQueue()
        
        req = create_mock_request("req1", priority=1)
        
        queue.add_request(req)
        assert len(queue) == 1
        
        queue.remove_request(req)
        assert len(queue) == 0
        
        # Re-add the same request
        queue.add_request(req)
        assert len(queue) == 1
        assert queue.pop_request() == req

    def test_same_priority_ordering(self):
        """Test that requests with same priority are ordered by arrival time."""
        queue = PriorityRequestQueue()
        
        # Same priority, different arrival times
        req1 = create_mock_request("req1", priority=1, arrival_time=3.0)
        req2 = create_mock_request("req2", priority=1, arrival_time=1.0)
        req3 = create_mock_request("req3", priority=1, arrival_time=2.0)
        
        queue.add_request(req1)
        queue.add_request(req2)
        queue.add_request(req3)
        
        # Should be ordered by arrival time when priority is the same
        assert queue.pop_request() == req2  # earliest arrival
        assert queue.pop_request() == req3
        assert queue.pop_request() == req1  # latest arrival


class TestPriorityQueuePerformance:
    """Performance tests for lazy deletion optimization."""

    def benchmark_remove_operations(self, queue_size: int, 
                                   remove_batch_size: int,
                                   num_iterations: int) -> dict:
        """Benchmark remove operations."""
        queue = PriorityRequestQueue()
        
        # Create and add requests
        requests = [
            create_mock_request(f"req{i}", priority=i % 10, arrival_time=float(i))
            for i in range(queue_size)
        ]
        
        for req in requests:
            queue.add_request(req)
        
        # Benchmark batch removals
        total_time = 0
        for _ in range(num_iterations):
            # Select random batch to remove
            batch = random.sample(requests[:len(queue)], 
                                min(remove_batch_size, len(queue)))
            
            start = time.perf_counter()
            queue.remove_requests(batch)
            total_time += time.perf_counter() - start
            
            # Re-add half of them
            for req in batch[::2]:
                queue.add_request(req)
        
        return {
            'total_time': total_time,
            'avg_time_per_batch': total_time / num_iterations,
            'queue_size': queue_size,
            'batch_size': remove_batch_size
        }

    @pytest.mark.benchmark
    def test_performance_small_queue(self):
        """Test performance with small queue (100 items)."""
        results = self.benchmark_remove_operations(
            queue_size=100,
            remove_batch_size=10,
            num_iterations=50
        )
        
        # Should complete quickly for small queues
        assert results['avg_time_per_batch'] < 0.001  # < 1ms

    @pytest.mark.benchmark
    def test_performance_large_queue(self):
        """Test performance with large queue (5000 items)."""
        results = self.benchmark_remove_operations(
            queue_size=5000,
            remove_batch_size=50,
            num_iterations=50
        )
        
        # Should still be fast even for large queues
        assert results['avg_time_per_batch'] < 0.01  # < 10ms

    @pytest.mark.benchmark
    def test_performance_vs_threshold(self):
        """Test how rebuild threshold affects performance."""
        thresholds = [0.3, 0.5, 0.7, 0.9]
        results = []
        
        for threshold in thresholds:
            queue = PriorityRequestQueue(rebuild_threshold=threshold)
            
            # Add many requests
            requests = [
                create_mock_request(f"req{i}", priority=i % 10)
                for i in range(1000)
            ]
            for req in requests:
                queue.add_request(req)
            
            # Remove most of them
            start = time.perf_counter()
            queue.remove_requests(requests[:700])  # Remove 70%
            elapsed = time.perf_counter() - start
            
            results.append((threshold, elapsed))
        
        # Higher thresholds should generally be faster
        # (fewer rebuilds, but more skipping in pop)
        # This is a trade-off that depends on workload
        for i in range(len(results) - 1):
            # Just verify it completes reasonably fast
            assert results[i][1] < 0.1  # < 100ms