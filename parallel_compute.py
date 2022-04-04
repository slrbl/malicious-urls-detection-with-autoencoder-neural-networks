"""
Ray Utilities

Ray is an open source project that makes it simple
to scale any compute-intensive Python workload.

These utility classes and functions enable parallel
execution of multiple instances of any function,
provided that data transferred between workers
and nodes can be serialized/deserialized
by Apache Arrow's Plasma object store.

Pipelining is used to maximise throughput
and an optional tqdm progressbar can be displayed.

Inspired by:
https://github.com/honnibal/spacy-ray/pull/1/files#diff-7ede881ddc3e8456b320afb958362b2aR12-R45
https://docs.ray.io/en/latest/auto_examples/progress_bar.html
"""
import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Awaitable, Optional

import ray
from ray.actor import ActorHandle
from tqdm import tqdm  # type: ignore


@ray.remote
class ProgressBarActor:
    """Utility class for Ray-compatible tqdm progressbar."""

    counter: int
    delta: int
    event: asyncio.Event

    def __init__(self) -> None:
        """Initialize progressbar actor."""
        self.counter = 0
        self.delta = 0
        self.event = asyncio.Event()

    def update(self, num_items_completed: int) -> None:
        """Update the progressbar with the incremental
        number of items that were just completed.

        Args:
            num_items_completed (int): Number of items that were just completed
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> tuple[int, int]:
        """Blocking call.

        Wait until somebody calls `update`, then return a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.

        Returns:
            tuple[int, int]: (Number of updates since
            the last call to `wait_for_update`,
            Total number of completed items)
        """
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter

    def get_counter(self) -> int:
        """Return the total number of complete items.

        Returns:
            int: Total number of complete items
        """
        return self.counter


class ProgressBar:
    """Ray-compatible tqdm progressbar."""

    progress_actor: ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, total: int, description: str = "") -> None:
        """Initialize progressbar.

        Ray actors don't seem to play nice with mypy, generating
        a spurious warning for the following line
        `self.progress_actor = ProgressBarActor.remote()`,
        which we need to suppress. The code is fine.

        Args:
            total (int): Total number of ticks in progressbar
            description (str, optional): Text description to display
            before progressbar in console. Defaults to "".
        """
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.total = total
        self.description = description

    @property
    def actor(self) -> ActorHandle:
        """Return a reference to the remote `ProgressBarActor`.

        When a task is completed, call `update` on the actor.

        Returns:
            ActorHandle: A reference to the remote `ProgressBarActor`
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method will return.

        See
        https://stackoverflow.com/questions/41707229/tqdm-printing-to-newline
        """

        pbar = tqdm(desc=self.description, total=self.total, position=0, leave=True)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return


@ray.remote
def run_task_handler(
    task_handler: Callable[..., Awaitable],
    task_args: tuple,
    object_store_ids: Mapping,
    actor_id: Optional[Any] = None,
) -> Any:
    """Runs `task_handler` on `task_args`,
    update progressbar and return the value returned by `task_handler`

    Reference
    https://docs.ray.io/en/latest/async_api.html#asyncio-for-remote-tasks

    Args:
        task_handler (Callable[...,Awaitable]): Asynchronous
        function to parallelise
        task_args (tuple): Arguments to be passed into `task_handler`
        object_store_ids (Mapping): Serializable object IDs to be passed
        from Ray object store into `task_handler`
        actor_id (Optional[Any], optional): Object reference assigned
        to `ProgressBar` actor. Defaults to None.

    Returns:
        Any: Value returned by `task_handler`
    """

    async def run_task_handler_(
        task_handler: Callable[..., Awaitable],
        task_args: tuple,
        object_store_ids: Mapping,
        actor_id: Optional[Any] = None,
    ) -> Any:
        result = await task_handler(*task_args, **{arg: await object_store_ids[arg] for arg in object_store_ids})

        if actor_id is not None:
            actor_id.update.remote(1)  # type: ignore
        return result

    return asyncio.get_event_loop().run_until_complete(run_task_handler_(task_handler, task_args, object_store_ids, actor_id))


def execute_with_ray(
    task_handler: Callable,
    task_args_list: Sequence[tuple],
    object_store: Optional[Mapping] = None,
    progress_bar: bool = True,
) -> list:
    """Apply task_handler to list of tasks.

    Tasks are processed in parallel with pipelining to maximise throughput.

    Args:
        task_handler (Callable): Asynchronous function to parallelise
        task_args_list (Sequence[tuple]): Sequence of tuples of Arguments
        to be passed into each `task_handler` instance
        object_store (Optional[Mapping], optional): Serializable objects,
        common to all task instances, to be put into
        Ray object store, if any. Defaults to None.
        progress_bar (bool, optional): If set to True, shows progressbar.
        Defaults to True.

    Returns:
        list: List of returned values from each instance of task_handler
    """

    if not task_args_list:
        return []

    if progress_bar:
        num_ticks = len(task_args_list)
        pbar = ProgressBar(num_ticks)
        actor = pbar.actor
        actor_id = ray.put(actor)

    # Put large serializable objects common to
    # all task instances into Ray object store
    object_store_ids: dict[str, Any] = {key: ray.put(object_store[key]) for key in object_store} if object_store else {}

    tasks_pre_launch: list[Awaitable] = [
        run_task_handler.remote(  # type:ignore
            task_handler,
            task_args,
            object_store_ids,
            actor_id=actor_id if progress_bar else None,
        )
        for task_args in task_args_list
    ]

    # Keeps progressbar open until all tasks are completed
    if progress_bar:
        pbar.print_until_done()

    # Processes tasks with pipelining
    results = []
    while tasks_pre_launch:
        done_id, tasks_pre_launch = ray.wait(tasks_pre_launch)
        results.append(ray.get(done_id[0]))

    return results
