import random

def bubble_sort(arr, counter=None):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if counter:
                counter.increment()
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr


def generate_data(size):
    return [random.randint(1, 10000) for _ in range(size)]
