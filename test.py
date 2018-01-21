from functools import cmp_to_key

x = [[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]

print(x)

def mycmp(a, b):
	return a[0] > b[0] or (a[0] == b[0] and a[1] < b[1])

x.sort(key=cmp_to_key(lambda (a, b):(-a, b)))

print(x)