import os
import random
import re



b = list(range(1, 101))
counter = 0
c = []
curMax = b[-1]
while len(b) > 1:
    d = 0
    while d < 7:
        counter += 1
        if counter > curMax:
            counter = counter - curMax
        if counter not in c:
            d += 1

    c.append(counter)
    b.remove(counter)
    curMax = max(b)

# print(c, b)
# for i in range(1, 101):
#     if i not in c:
#         print(i)


def QuickSort(lList, start, end):
    if start >= end:
        return
    flag = lList[end]
    i, j = start, end - 1
    while i <= j:
        if lList[i] > flag:
            lList[j], lList[i] = lList[i], lList[j]
            j -= 1
        else:
            i += 1
    lList[j+1], lList[end] = lList[end], lList[j+1]
    print(lList, flag)
    QuickSort(lList, start, j)
    QuickSort(lList, j+2, end)

# a = list(range(70))
# random.shuffle(a)
# print(a)
# QuickSort(a, 0, len(a)-1)
# print(a)


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
        return len(nums) - is_same.index(False) - is_same[::-1].index(False)


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
    from functools import cmp_to_key
    l = []
    for each in words:
        tmp = [each, words.count(each)]
        if tmp not in l:
            l.append(tmp)    
    
    l.sort(key=lambda x:x[0])
    l.sort(key=lambda x:x[1], reverse=True)
    return [m[0] for m in l[:k]]

print(topKFrequent(["aa","aaa","a"], 1))