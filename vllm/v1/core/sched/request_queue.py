# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""
    FCFS = "fcfs"
    PRIORITY = "priority"


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass

    @abstractmethod
    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        self.extendleft(reversed(requests))

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [
            req for req in self if req not in requests_to_remove
        ]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        return super().__reversed__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue with lazy deletion optimization for heap operations.

    Requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    
    This implementation uses lazy deletion to optimize remove operations:
    - Removing items marks them as deleted instead of rebuilding the heap
    - Deleted items are skipped during pop operations
    - The heap is rebuilt when deleted items exceed a threshold
    """

    def __init__(self, rebuild_threshold: float = 0.5) -> None:
        """Initialize the priority queue.
        
        Args:
            rebuild_threshold: Fraction of deleted items that triggers rebuild.
                              Default 0.5 (rebuild when 50% of heap is deleted).
        """
        self._heap: list[tuple[int, float, Request]] = []
        self._deleted: set[Request] = set()  # Track deleted requests
        self._size = 0  # Actual number of valid items
        self._rebuild_threshold = rebuild_threshold

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        # If this request was previously deleted, remove from deleted set
        self._deleted.discard(request)
        heapq.heappush(self._heap,
                       (request.priority, request.arrival_time, request))
        self._size += 1

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        # Skip deleted items lazily
        while self._heap:
            priority, arrival, request = heapq.heappop(self._heap)
            if request not in self._deleted:
                self._size -= 1
                return request
            else:
                # Remove from deleted set as we've now removed it from heap
                self._deleted.discard(request)
        raise IndexError("pop from empty heap")

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        # Skip deleted items at the front
        while self._heap:
            _, _, request = self._heap[0]
            if request not in self._deleted:
                return request
            # Remove deleted item from front
            heapq.heappop(self._heap)
            self._deleted.discard(request)
        raise IndexError("peek from empty heap")

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.
        
        Note: In a priority queue, there is no concept of prepending to the 
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.
        
        Note: In a priority queue, there is no concept of prepending to the 
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue using lazy deletion."""
        if request not in self._deleted:
            self._deleted.add(request)
            self._size -= 1
            self._maybe_rebuild()

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests using lazy deletion.
        
        This is O(k) where k is the number of requests to remove,
        instead of O(n) for rebuilding the entire heap.
        """
        for request in requests:
            if request not in self._deleted:
                self._deleted.add(request)
                self._size -= 1
        self._maybe_rebuild()

    def _maybe_rebuild(self) -> None:
        """Rebuild the heap if too many items are deleted."""
        if len(self._deleted) > len(self._heap) * self._rebuild_threshold:
            self._rebuild()

    def _rebuild(self) -> None:
        """Rebuild the heap without deleted items."""
        old_heap = self._heap
        self._heap = []
        for priority, arrival, request in old_heap:
            if request not in self._deleted:
                heapq.heappush(self._heap, (priority, arrival, request))
        self._deleted.clear()

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return self._size > 0

    def __len__(self) -> int:
        """Get number of valid (non-deleted) requests in queue."""
        return self._size

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            _, _, request = heapq.heappop(heap_copy)
            if request not in self._deleted:
                yield request

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse priority order."""
        return reversed(list(self))


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
