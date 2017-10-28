import os
import random
import re
import math

import collections


def findDisappearedNumbers(nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        for each in nums:
            each = abs(each) - 1
            nums[each] = -abs(nums[each])

        return (i+1 for i, each in enumerate(nums) if each > 0)

# l = [4,3,2,7,8,2,3,1]
# result = findDisappearedNumbers(l)
# result = list(result)
# print(result)

def moveZeroes(nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        x = 0
        for i, each in enumerate(nums):
            if each != 0:
                nums[x], nums[i] = nums[i], nums[x]
                x += 1

# l = [0, 1, 0, 3, 0, 0, 0, 1, 2, 3, 12]
# result = moveZeroes(l)
# print(l)


def twoSum(numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        # for i, each in enumerate(numbers):
        #     r = len(numbers) - 1
        #     l = i + 1
        #     tar = target - each
        #     while l <= r:
        #         mid = l + (r-l)//2
        #         if numbers[mid] == tar:
        #             return [i+1, mid+1]
        #         elif numbers[mid] < tar:
        #             l = mid + 1
        #         else:
        #             r = mid - 1
        l, r = 0, len(numbers) - 1
        while l < r:
            lr = numbers[l] + numbers[r]
            if lr == target:
                return [l+1, r+1]
            elif lr < target:
                l += 1
            else:
                r -= 1


# l = [2, 7, 3, 5, 34, 8, 12, 43, 11, 15];target = 54
# result = twoSum(l, target)
# print(result)

def majorityElement(nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        count = 0
        for each in nums:
            if count == 0:
                major = each
                count += 1
            elif major == each:
                count += 1
            else:
                count -= 1
        return major

# l = [2, 7, 3, 5, 34, 3, 3, 3, 11, 15]
# result = majorityElement(l)
# print(result)

def containsDuplicate(nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        nums2 = set(nums)
        return len(nums) != len(nums2)

# l = [3, 1]
# result = containsDuplicate(l)
# print(result)


def maximumProduct(nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        s2 = nums[-2] * nums[-3]
        if nums[-1] > 0:
            s1 = nums[0] * nums[1]
            s = max(s1, s2)
            return s*nums[-1]
        else:
            return s2 * nums[-1]

        # 简洁版
        # return max(nums[-1] * nums[-2] * nums[-3], nums[0] * nums[1] * nums[-1])

# l = [-3, -1, -3, -4, -7, -8]
# result = maximumProduct(l)
# print(result)

def missingNumber(nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        return (n+1)*n/2 - sum(nums)


def maxProfit(prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        # minp = float('inf')
        # maxp = 0
        # for each in prices:
        #     minp = min(each, minp)
        #     profit = each - minp
        #     maxp = max(profit, maxp)
        # return maxp

        # 第二种
        curmax, allmax = 0, 0
        nLen = len(prices)
        for i in range(1, nLen):
            curmax += prices[i] - prices[i-1]
            curmax = max(0, curmax)
            allmax = max(allmax, curmax)
        return allmax

# l = [-2, -1, -3]
# result = maxProfit(l)
# print(result)

def maxSubArray(nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        curmax, allmax = 0, 0
        nLen = len(nums)
        for each in nums:
            curmax += each
            curmax = max(0, curmax)
            allmax = max(allmax, curmax)
        return allmax or max(nums)


def searchInsert(nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        l, r = 0, len(nums)-1
        while l <= r:
            mid = (r+l)//2
            if nums[mid] > target:
                r = mid - 1
            elif nums[mid] < target:
                l = mid + 1
            else:
                return mid, l, r
        return l, l, r

# l = [1, 3, 6, 8, 11, 13, 15]
# result = searchInsert(l, 9)
# print(result)

def generate(numRows):
    #生成杨辉三角
    """
    :type numRows: int
    :rtype: List[List[int]]
    """
    l = []
    for i in range(numRows):
        if i == 0:
            l.append([1])
        else:
            lTmp = [1]
            lLast = l[-1]
            for j, each in enumerate(lLast[:-1]):
                lTmp.append(each+lLast[j+1])
            lTmp.append(1)
            l.append(lTmp)
    return l

    # 更佳解
    # res = [[1]]
    #     for i in range(1, numRows):
    #         res += [map(lambda x, y: x+y, res[-1] + [0], [0] + res[-1])]
    #     return res[:numRows]

    # explanation: Any row can be constructed using the offset sum of the previous row. Example:

    #     1 3 3 1 0
    #  +  0 1 3 3 1
    #  =  1 4 6 4 1

# result = generate(10)
# for each in result:
#     print(each)


def removeElement(nums, val):
    """
    :type nums: List[int]
    :type val: int
    :rtype: int
    """
    popped = 0
    for i in range(len(nums)):
        i -= popped
        if nums[i] == val:
            nums.pop(i)
            popped += 1

    return nums


# l = [1, 3, 6, 8, 3, 3, 11, 13, 3]
# result = removeElement(l, 3)
# print(result)

def findMaxAverage(nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: float
    """
    cursum, allmax = 0, 0
    nLen = len(nums)
    for i in range(k):
        cursum += nums[i]
    allmax = cursum
    for i in range(k, nLen):
        cursum = cursum - nums[i-k] + nums[i]
        allmax = max(cursum, allmax)

    return allmax/float(k)


def canPlaceFlowers(flowerbed, n):
    """
    :type flowerbed: List[int]
    :type n: int
    :rtype: bool
    """
    nLen = len(flowerbed)
    m = 0
    for i, each in enumerate(flowerbed):
        if each == 0 and (i==0 or flowerbed[i-1]==0) and (i==nLen-1 or flowerbed[i+1]==0):
            flowerbed[i] = 1
            m += 1
    return m


# l = [0,0,0,0,1,0,0,0,0,0,1,0,0,0,0]
# result = canPlaceFlowers(l, 0)
# print(result)


def findUnsortedSubarray(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    is_same = [a == b for a, b in zip(nums, sorted(nums))]
    if all(is_same):
        return 0
    else:
        return is_same.rindex(False) - is_same.index(False)


# l = [2, 6, 4, 8, 10, 9, 15]
# result = findUnsortedSubarray(l)
# print(result)

def checkPossibility(nums):
    #改变一个数 使整体有序
    """
    :type nums: List[int]
    :rtype: bool
    """
    a, b = list(nums), list(nums)
    for i in range(len(nums)-1):
        if nums[i] > nums[i+1]:
            a[i] = nums[i+1]
            b[i+1] = nums[i]
            break
    return a == sorted(a) or b == sorted(b)


def maxAreaOfIsland(grid):
    #围棋一片同气的棋子数 最大值
    """
    :type grid: List[List[int]]
    :rtype: int
    """    
    grid = {i + j*1j: val for i, row in enumerate(grid) for j, val in enumerate(row)}
    def area(z):
        return grid.pop(z, 0) and 1 + sum(area(z + 1j**k) for k in range(4))
    return max(map(area, set(grid)))
        

# result = maxAreaOfIsland(grid)
# print(result)


def findPoisonedDuration(timeSeries, duration):
    #提莫攻击
    """
    :type timeSeries: List[int]
    :type duration: int
    :rtype: int
    """
    lasttime = duration if timeSeries else 0
    for i in range(len(timeSeries)-1):
        diff = timeSeries[i+1] - timeSeries[i]
        lasttime += min(diff, duration)
    
    return lasttime

# l = [1,2, 4, 8]
# result = findPoisonedDuration(l, 4)
# print(result)

def constructArray(n, k):
    #使n个数的数组的相邻差只有k种值
    """
    :type n: int
    :type k: int
    :rtype: List[int]
    """
    res = []
    l, r = 1, k+1
    while l <= r:
        res.append(l)
        l+=1
        if l <= r :
            res.append(r)
            r-=1
    
    res.extend(range(k+2, n+1))
    return res

# print(constructArray(20,10))


def topKFrequent(words, k):
    """
    :type words: List[str]
    :type k: int
    :rtype: List[str]
    """
    import collections as clt
    oCount = clt.Counter(words)
    l = []
    for key, val in oCount.items():
        l.append([key, val])    

    # l = []
    # for each in words:
    #     tmp = [each, words.count(each)]
    #     if tmp not in l:
    #         l.append(tmp)    
    
    l.sort(key=lambda x:x[0])
    l.sort(key=lambda x:x[1], reverse=True)
    return [m[0] for m in l[:k]]

# print(topKFrequent(["aa","aaa","a"], 1))

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None



def reverseList(head):
    #翻转单向链表
    """
    :type head: ListNode
    :rtype: ListNode
    """
    prev = None
    while head:
        cur = head
        head = head.next
        cur.next = prev
        prev = cur
    return prev



def updateMatrix(matrix):
    #矩阵每个值与0的距离
    """
    :type matrix: List[List[int]]
    :rtype: List[List[int]]
    """
    w, h = len(matrix[0]), len(matrix)
    inf = float('inf')
    lZero = []
    for i in range(h):
        for j in range(w):
            if matrix[i][j]:
                matrix[i][j] = inf
            else:
                lZero.append((i, j))
    
    for i, j in lZero:
        tmp = matrix[i][j] + 1
        for r, c in [(i+1,j), (i-1,j), (i,j+1), (i,j-1)]:
            if 0 <= r < h and 0 <= c < w and matrix[r][c] > tmp:
                matrix[r][c] = tmp
                lZero.append((r,c))
    
    return matrix



l= [[0,1,1,1,1],
    [1,1,1,1,0],
    [1,1,1,1,1]]
# print(updateMatrix(l))

def minWindow(s, t):
    #包含指定字符的最小子串

    need, missing = collections.Counter(t), len(t)
    l = missing
    i = I = J = 0
    for j, c in enumerate(s, 1):
        missing -= need[c] > 0
        need[c] -= 1
        if not missing:
            while i < j and need[s[i]] < 0:
                need[s[i]] += 1
                i += 1
            if not J or j - i <= J - I:
                I, J = i, j
                # if J-I == l:
                #     break
    return s[I:J]


# print(minWindow("CBADADOBECODEBANC", "ABC"))


def detectCycle(head):
    #检测链表是否有环并找出起点
    """
    :type head: ListNode
    :rtype: ListNode
    """
    p1 = p2 = head
    try:
        p1 = p1.next
        p2 = p2.next.next
        while p1 != p2:
            p1 = p1.next
            p2 = p2.next.next
    except:
        return None
    meet = p1
        
    while head != meet:
        head = head.next
        meet = meet.next
        
    return head

def checkPerfectNumber(num):
    """
    :type num: int
    :rtype: bool
    """
    import math
    s = 1
    isqrt = int(math.sqrt(num))
    for i in range(2, isqrt+1):
        if num % i == 0:
            print(i, num // i)
            s += i
            if num / i != i:
                s += num // i
                
    return s == num

# print(checkPerfectNumber(200))

def makeresult(l):
    tmp = []
    for i in range(n):
        tmp.append(["."]*n)
    for i, j in enumerate(l):
        tmp[i][j] = "Q"
    for i, each in enumerate(tmp):
        tmp[i] = "".join(each)
    return tmp
        
def backtracking(l, i):
    if i == n:
        lAll.append(makeresult(l))
        return
    for j in range(n):
        l[i], bValid = j, True
        for r in range(i):
            if l[r] == j or abs(l[r]-j) == i - r:
                bValid = False
                break
        if bValid:
            backtracking(l,i+1)

# n=4            
# lAll = []                
# l = [0]*n
# backtracking(l,0)

# print(lAll, len(lAll))



# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


def rob(root):
    """
    :type root: TreeNode
    :rtype: int
    """
    
    def robdfs(root):
        if not root:
            return [0, 0]
        l = robdfs(root.left)
        r = robdfs(root.right)
        
        return [max(l[0],l[1])+max(r[0],r[1]), root.val+l[0]+r[0]]
    
    return max(robdfs(root))


def isSubtree(s, t):
    #二叉树
    """
    :type s: TreeNode
    :type t: TreeNode
    :rtype: bool
    """
    l = []
    
    def GetDepth(node, iCheck=None):
        ld = GetDepth(node.left, iCheck) if node.left else 0
        rd = GetDepth(node.right, iCheck) if node.right else 0
        depth = 1 + max(ld, rd)
        if iCheck!=None and depth == iCheck:
            l.append(isSub(node, t))
        return depth

    def isSub(node1, node2):
        if not node1 or not node2:
            return not node1 and not node2
        
        return node1.val == node2.val and isSub(node1.left, node2.left) and isSub(node1.right, node2.right)
    
    tdepth = GetDepth(t)
    GetDepth(s,tdepth)
    return any(l)

def findKthNumber(m, n, k):
    # m x n乘法表里的第k小的数
    """
    :type m: int
    :type n: int
    :type k: int
    :rtype: int
    """
    l, r = 1, m*n
    while l < r:
        mid = (l+r) // 2
        if sum(min(mid//i, n) for i in range(1,m+1)) < k:
            l = mid + 1
        else:
            r = mid
    
    return l


def mergeTrees(t1, t2):
    """
    :type t1: TreeNode
    :type t2: TreeNode
    :rtype: TreeNode
    """
    if not t1 and not t2:
        return None
    
    val1 = t1.val if t1 else 0
    val2 = t2.val if t2 else 0
    
    oNew = TreeNode(val1+val2)
    
    oleft1 = t1.left if t1 else None
    oleft2 = t2.left if t2 else None
    oright1 = t1.right if t1 else None
    oright2 = t2.right if t2 else None
    
    oNew.left = self.mergeTrees(oleft1, oleft2)
    oNew.right = self.mergeTrees(oright1, oright2)
    
    return oNew



def islandPerimeter(grid):
    #岛的周长
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    
    def cal(lGrid):
        s = 0
        for each in lGrid:
            each = [0] + each + [0]
            for i in range(len(each)-1):                    
                if each[i] != each[i+1]:
                    s += 1
        return s
    
    
    d = 0
    d += cal(grid)
    symmetry = map(list, zip(*grid))
    d += cal(symmetry)
    
    return d



def climbStairs(n):
    if n in (1,2):
        return n
    
    a,b=1,2
    s = 0
    for i in range(n-2):
        s = a+b
        a,b=b,s
    return s

# print(climbStairs(35))


def isValid(s):
    """
    :type s: str
    :rtype: bool
    """
    l = []
    for each in s:
        if each in ('(', '[', '{'):
            l.append(each)
        elif each == ')':
            if l.pop(-1) != '(':
                return False
        elif each == ']':
            if l.pop(-1) != '[':
                return False
        elif each == '}':
            if l.pop(-1) != '{':
                return False
    return True

# print(isValid("[([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([([()])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])])]"))


def findAnagrams(s, p):
    """
    :type s: str
    :type p: str
    :rtype: List[int]
    """
    len1 = len(s)
    len2 = len(p)
    if len1 < len2:
        return []
    
    result = []
    oCounter = collections.Counter(p)
    tmp = list(s[:len2])
    for each in tmp:
        if each in oCounter:
            oCounter[each] -= 1
    
    if not any(oCounter.values()):
        result.append(0)
    for i in range(len2, len1):
        c = tmp.pop(0)
        if c in oCounter:
            oCounter[c] += 1
        tmp.append(s[i])
        if s[i] in oCounter:
            oCounter[s[i]] -= 1
        if not any(oCounter.values()):
            result.append(i-len2+1)
    
    return result

# print(findAnagrams("ab", "abc"))


def isPalindrome(head):
    #链表值是否构成回文
    """
    :type head: ListNode
    :rtype: bool
    """
    length = 0
    node = head
    while node:
        node = node.next
        length += 1
    
    node = head
    for _ in range((length+1)//2):
        node = node.next
    
    cur = prev = None
    while node:
        cur = node
        node = node.next
        cur.next = prev
        prev = cur
    
    while cur and head:
        if cur.val != head.val:
            return False
        cur = cur.next
        head = head.next
        
    return True

    # rev = None
    # fast = head
    # while fast and fast.next:
    #     fast = fast.next.next
    #     rev, rev.next, head = head, rev, head.next
    # tail = head.next if fast else head
    # isPali = True
    # while rev:
    #     isPali = isPali and rev.val == tail.val
    #     head, head.next, rev = rev, head, rev.next
    #     tail = tail.next
    # return isPali

def lengthOfLongestSubstring(s):
    """
    :type s: str
    :rtype: int
    """
    d = {}
    i=j=I=J=0
    for each in s:
        d[each] = d.get(each, 0) + 1
        j += 1
        if d[each] == 2:
            while s[i] != each:
                d[s[i]] -= 1
                i += 1
            d[s[i]] -= 1
            i += 1
            continue
        if j-i >= J-I:
            I, J = i, j
    
    return s[I:J], J-I

print(lengthOfLongestSubstring("zxcvawefeasdfzv"))