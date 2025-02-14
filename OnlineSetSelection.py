#!/usr/bin/env python
import itertools
from typing import List
import math
import heapq
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


class Item:
    def __init__(self, name: str, category: int, score: float):
        self.name = name
        self.category = category
        self.score = score

    def __str__(self):
        return f"name={self.name}, category={self.category}, score={self.score}"

    def __lt__(self, other):
        return self.score < other.score


class BoundedHeap:
    def __init__(self, max_size):
        self.heap = []
        self.max_size = max_size

    def offer(self, x):
        if self.max_size <= 0:
            return
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, x)
        else:
            min_item = self.get_min_element()
            if x > min_item:
                heapq.heappop(self.heap)
                heapq.heappush(self.heap, x)

    def delete_min_element(self):
        if self.heap:
            return heapq.heappop(self.heap)

    def get_min_element(self):
        return self.heap[0] if self.heap else Item(None, None, float("-inf"))


def algo1(I: List[Item], K: int, d: int, floors: List[int], ceilings: List[int]) -> List[Item]:
    """
    Diverse top-k selection from a sorted list.
    Args:
        I:          List of items sorted by score
        K:          Number of items to select
        d:          Number of categories
        floors:     Constraints floor_i ≤ k_i for each i ∈ [1. . .d].
        ceilings:   Constraints k_i ≤ ceil_i for each i ∈ [1. . .d].

    Returns:
        L top K  chosen items from list I.
    """
    L = []
    C = [0] * d
    slack = K - sum(floors)
    iterator = iter(I)
    while len(L) < K:
        x = next(iterator)
        i = x.category
        if C[i] < floors[i]:
            L.append(x)
            C[i] += 1
        elif (C[i] < ceilings[i]) and (slack > 0):
            L.append(x)
            C[i] += 1
            slack -= 1
    return L


def algo2(I: List[Item], K: int, d: int, floors: List[int], ceilings: List[int], items_per_category: List[int],
          warm_up_factor=1) -> List[Item]:
    """
    Diverse top-k selection from a sorted list.
    Args:
        I:                  Stream of items.
        K:                  Number of items to select
        d:                  Number of categories
        floors:             Constraints floor_i ≤ k_i for each i ∈ [1. . .d].
        ceilings:           Constraints k_i ≤ ceil_i for each i ∈ [1. . .d].
        items_per_category: n_i for i ∈[1 . . .d].
        warm_up_factor      multi the warmup number of elements

    Returns:
        L top K  chosen items from list I.
    """

    N = len(I)
    L = []
    C = [0] * d
    M = [0] * d
    R = [math.floor((n / math.e) * warm_up_factor) for n in items_per_category]
    T_i = [BoundedHeap(floor) for floor in floors]
    slack = K - sum(floors)

    r = math.floor((N / math.e) * warm_up_factor)
    T = BoundedHeap(slack)
    iterator = iter(I)
    num_feasible_items = lambda: sum(
        items_per_category[i] - M[i] for i in range(len(items_per_category)) if ceilings[i] - C[i] > 0)
    while len(L) < K:
        x = next(iterator)
        i = x.category
        sum_m = sum(M)
        if sum_m < r:
            T.offer(x)
        if M[i] < R[i]:
            T_i[i].offer(x)
        elif ((C[i] < floors[i]) and (x.score > T_i[i].get_min_element().score)) or items_per_category[i] - M[i] == \
                floors[i] - C[i]:
            T_i[i].delete_min_element()
            L.append(x)
            C[i] += 1
        elif (sum_m >= r) and (x.score > T.get_min_element().score and (C[i] < ceilings[i]) and (slack > 0)):
            T.delete_min_element()
            L.append(x)
            C[i] += 1
            slack -= 1
        elif (C[i] < ceilings[i]) and (num_feasible_items() == K - len(L)):
            L.append(x)
            C[i] += 1
            slack -= 1
        M[i] += 1
    return L


def algo3(I: List[Item], K: int, d: int, floors: List[int], ceilings: List[int], items_per_category: List[int],
          warm_up_factor=1) \
        -> List[Item]:
    """
    Diverse top-k selection from a sorted list.
    Args:
        I:                  Stream of items.
        K:                  Number of items to select
        d:                  Number of categories
        floors:             Constraints floor_i ≤ k_i for each i ∈ [1. . .d].
        ceilings:           Constraints k_i ≤ ceil_i for each i ∈ [1. . .d].
        items_per_category: n_i for i ∈[1 . . .d].
        warm_up_factor      multi the warmup number of elements

    Returns:
        L top K  chosen items from list I.
    """
    D = [BoundedHeap(ceil) for ceil in ceilings]
    C = [0] * d
    M = [0] * d
    R = [math.floor((n / math.e) * warm_up_factor) for n in items_per_category]
    T_i = [BoundedHeap(floor) for floor in floors]
    u = d - sum(1 for floor in floors if floor == 0)
    w = 0
    iterator = iter(I)
    while u > 0 or w < K:
        try:
            x = next(iterator)
        except StopIteration as e:
            # No more items
            break
        i = x.category
        if M[i] < R[i]:
            T_i[i].offer(x)
        elif (C[i] < ceilings[i]) and (x.score > T_i[i].get_min_element().score):
            C[i] += 1
            T_i[i].delete_min_element()
            if (floors[i] > 0) and C[i] == floors[i]:
                u -= 1
        D[i].offer(x)
        M[i] += 1
        w = sum(len(d_i.heap) for d_i in D)
    W = BoundedHeap(w)
    for d_i in D:
        for item in d_i.heap:
            W.offer(item)
    W.heap.sort(key=lambda x: x.score, reverse=True)
    L = algo1(W.heap, K, d, floors, ceilings)
    return L


df = pd.read_csv("2024 Billionaire List.csv")
df = df.head(400)
df['2024 Net Worth'] = (
    df['2024 Net Worth']
    .str.replace('$', '', regex=False)  # Remove the dollar sign
    .str.replace('B', '', regex=False)  # Remove the "B"
    .astype(float)  # Convert billions to integers
)

items = []
gender = {'M': 0, 'F': 1}
for index, row in df.iterrows():
    items.append(Item(row['Name'], gender[row["Gender"]], row['2024 Net Worth']))

selected_items = algo1(items, 4, len(gender), [2, 2], [2, 2])
benchmark = sum(item.score for item in selected_items)

gender_counts = df['Gender'].value_counts()


def plot(algo, warmup_factor, color):
    warmup_label = {
        1: "1",
        0.25: "1/4",
        1 / 16: "1/16"
    }.get(warmup_factor, str(warmup_factor))
    walking_distances = []
    scores = []
    K = 4
    N = len(items)

    for i in range(400):
        random.shuffle(items)  # Shuffle items
        selected_items = algo(items, K, len(gender), [2, 2], [2, 2], [gender_counts["M"], gender_counts["F"]],
                              warmup_factor)
        total_score = sum(item.score for item in selected_items)

        # Walking distance: index of the last selected item in the shuffled list
        last_item_index = max(items.index(item) for item in selected_items)
        walking_distances.append(last_item_index)
        scores.append(total_score)
    # Calculate the accuracy ratio
    ratio = [score / benchmark for score in scores]

    # Plot the results with a regression line and confidence interval
    plt.figure(figsize=(8, 6))
    sns.regplot(
        x=walking_distances,
        y=ratio,
        ci=60,
        scatter_kws={'alpha': 0.8, 'color': color},  # Apply color to scatter points
        line_kws={'color': "black"}  # Apply color to the regression line
    )

    # Customize plot appearance
    plt.xlim(0, N)  # Set x-axis from 0 to N
    plt.ylim(0, 1)  # Set y-axis from 0 to 1
    plt.xlabel("Walking distance (index of last selected item)")
    plt.ylabel("Overall accuracy (ratio to benchmark)")
    plt.title(f"{algo.__name__} Performance: Walking Distance vs. Accuracy (warm-up factor={warmup_label})")
    plt.savefig(os.path.join("plots", f"{algo.__name__}_warm_up_{warm_up_factor}.jpg"))


# Example usage
colors = ['#1f77b4',  # Blue
          '#ff7f0e',  # Orange
          '#2ca02c',  # Green
          '#d62728',  # Red
          '#9467bd',  # Purple
          '#8c564b']  # Brown
warm_up_factors = [1, 1 / 4, 1 / 16]
algos = (algo2, algo3)
for (algo, warm_up_factor), color in zip(itertools.product(algos, warm_up_factors), colors):
    plot(algo, warm_up_factor, color)
