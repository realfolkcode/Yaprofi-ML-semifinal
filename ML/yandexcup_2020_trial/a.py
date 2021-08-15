import math

arr = []
with open('stump.in', 'r') as f:
    for line in f.readlines()[1:]:
        x, y = map(float, line.strip().split())
        arr.append([x, y])
n = len(arr)
arr.sort()
s = 0.
for i in range(n):
    s += arr[i][1]
k = 0.
t = 0.
a_best = 0.
b_best = s / n
c_best = arr[0][0]
E_best = -2 * b_best * s + n * b_best**2
for i in range(n - 1):
    k = i + 1
    c = (arr[i][0] + arr[i+1][0]) / 2
    t += arr[i][1]
    if arr[i][0] == arr[i+1][0]:
        continue
    a = t / k
    b = (s - t) / (n - k)
    E = -(2 * a * t) - (2 * b * (s - t)) + k * a**2 + (n - k) * b**2
    if E < E_best:
        a_best = a
        b_best = b
        c_best = c
        E_best = E
with open('stump.out', "w") as f:
    f.write(str(a_best) + ' ' + str(b_best) + ' ' + str(c_best))
