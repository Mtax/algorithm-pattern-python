1. [二叉树](#二叉树)
   1. [知识点](#知识点)
      1. [二叉树的实现](#二叉树的实现)
      2. [二叉树遍历](#二叉树遍历)
         1. [递归模板](#递归模板)
         2. [前序非递归](#前序非递归)
         3. [中序非递归](#中序非递归)
         4. [后序非递归](#后序非递归)
         5. [BFS 层次遍历](#bfs-层次遍历)
      3. [分治法应用](#分治法应用)
   2. [常见题目示例](#常见题目示例)
      1. [maximum-depth-of-binary-tree](#maximum-depth-of-binary-tree)
      2. [balanced-binary-tree](#balanced-binary-tree)
      3. [binary-tree-maximum-path-sum](#binary-tree-maximum-path-sum)
      4. [lowest-common-ancestor-of-a-binary-tree](#lowest-common-ancestor-of-a-binary-tree)
      5. [BFS 层次应用](#bfs-层次应用)
      6. [binary-tree-zigzag-level-order-traversal](#binary-tree-zigzag-level-order-traversal)
      7. [二叉搜索树应用](#二叉搜索树应用)
      8. [validate-binary-search-tree](#validate-binary-search-tree)
         1. [insert-into-a-binary-search-tree](#insert-into-a-binary-search-tree)
   3. [总结](#总结)
   4. [练习](#练习)

# 二叉树
二叉树是一种非常重要的数据结构。

## 知识点

### 二叉树的实现


```python
"""Implementation of a Binary Tree.
"""


class Node:
    """The Node Class defines the structure of a Node"""

    def __init__(self, value: int) -> None:
        """Initialize Node with value.

        Args:
            value: value of current node.

        examples:
        >>> root = Node(10)
        >>> print(root)
        10
        >>> root.left = Node(5)
        >>> root.right = Node(3)

        # example Tree Structure
        #          10
        #        /    \
        #        5      3
        #     /    \  /    \
        #  None  None None None
        """
        self.root_value = value
        self.left = None
        self.right = None

    def __str__(self) -> str:
        return str(self.root_value)

    def insert(self, value: int):
        """Insert value into node.

        Args:
            value (int): value to be inserted.

        examples:
        >>> tree = Node(10)
        >>> print(tree)
        10
        >>> tree.insert(5)
        >>> print(tree.left)
        5
        >>> tree.insert(12)
        >>> print(tree.right)
        12
        >>> tree.insert(7)
        >>> print(tree.left.right)
        7
        >>> tree.insert(11)
        >>> print(tree.right.left)
        11

        # example Tree Structure
        #          10
        #        /    \
        #        5      12
        #     /    \  /    \
        #  None    7 11 None
        """
        if not self.root_value:
            self.root_value = value
            return

        # if value < root_value insert in left node
        if value < self.root_value:
            if not self.left:
                self.left = Node(value)
            else:
                self.left.insert(value)
        else:
            if not self.right:
                self.right = Node(value)
            else:
                self.right.insert(value)

```
### 二叉树遍历

树的遍历指的是按照某种规则，不重复地访问树的所有节点的过程。

根据访问节点的顺序不同，树的遍历可以分为深度优先遍历和广度优先遍历。深度优先遍历可细分为前序、中序以及后序遍历。

* 深度优先遍历
  * 前序遍历：**先访问根节点**，再前序遍历左子树，再前序遍历右子树
  * 中序遍历：先中序遍历左子树，**再访问根节点**，再中序遍历右子树
  * 后序遍历：先后序遍历左子树，再后序遍历右子树，**再访问根节点**
* 广度优先遍历（也称层次遍历）

树的定义是递归定义，因此用递归实现树的三种（前序、中序、后序）遍历容易理解且代码简洁。

> 相对来说深度优先应用更广泛，实现更简单，因此先不细讲广度优先遍历。

#### 递归模板

- 递归实现二叉树遍历非常简单，不同顺序区别仅在于访问父结点顺序

```Python
def preoder_traversal(node: Node):
    """In a preorder traversal, the root node is visited first,
    followed by the left child, then the right child.

    Args:
        node (Node): Node to be traversaled.

    examples:
    >>> tree = Node(10)
    >>> _ = [tree.insert(v) for v in [5, 12, 7, 11]]
    >>> print(tree)
    10
    >>> print(tree.left)
    5
    >>> print(tree.right)
    12
    >>> print(tree.left.right)
    7
    >>> print(tree.right.left)
    11
    >>> preoder_traversal(tree)
    10
    5
    7
    12
    11

    # example Tree Structure
    #          10
    #        /    \
    #        5      12
    #     /    \  /    \
    #  None    7 11   None

    """
    # import ipdb;ipdb.set_trace()
    if node is None:
        return
    print(node)
    preoder_traversal(node.left)
    preoder_traversal(node.right)


def inoder_traversal(node: Node):
    """In an inorder traversal, the left child is visited first,
    followed by the parent node, then followed by the right child.

    Args:
        node (Node): Node to be traversaled.

    examples:
    >>> tree = Node(10)
    >>> _ = [tree.insert(v) for v in [5, 12, 7, 11]]
    >>> inoder_traversal(tree)
    5
    7
    10
    11
    12

    # example Tree Structure
    #          10
    #        /    \
    #        5      12
    #     /    \  /    \
    #  None    7 11   None

    """
    if node is None:
        return
    inoder_traversal(node.left)
    print(node)
    inoder_traversal(node.right)


def postoder_traversal(node: Node):
    """In a postorder traversal, the left child is visited first, 
    followed by the right child, then the root node.

    Args:
        node (Node): Node to be traversaled.

    examples:
    >>> tree = Node(10)
    >>> _ = [tree.insert(v) for v in [5, 12, 7, 11]]
    >>> postoder_traversal(tree)
    7
    5
    11
    12
    10

    # example Tree Structure
    #          10
    #        /    \
    #        5      12
    #     /    \  /    \
    #  None    7 11   None

    """
    if node is None:
        return
    postoder_traversal(node.left)
    postoder_traversal(node.right)
    print(node)
```

#### [前序非递归](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)

- 本质上是图的 DFS 的一个特例，因此可以用栈来实现

```Python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        
        preorder = []
        if root is None:
            return preorder
        
        s = [root]
        while len(s) > 0:
            node = s.pop()
            preorder.append(node.val)
            if node.right is not None:
                s.append(node.right)
            if node.left is not None:
                s.append(node.left)
        
        return preorder
```

#### [中序非递归](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)

```Python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        s, inorder = [], []
        node = root
        while len(s) > 0 or node is not None:
            if node is not None:
                s.append(node)
                node = node.left
            else:
                node = s.pop()
                inorder.append(node.val)
                node = node.right
        return inorder
```

#### [后序非递归](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/)

```Python
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:

        s, postorder = [], []
        node, last_visit = root, None
        
        while len(s) > 0 or node is not None:
            if node is not None:
                s.append(node)
                node = node.left
            else:
                peek = s[-1]
                if peek.right is not None and last_visit != peek.right:
                    node = peek.right
                else:
                    last_visit = s.pop()
                    postorder.append(last_visit.val)
        
        
        return postorder
```

注意点

- 核心就是：根节点必须在右节点弹出之后，再弹出

DFS 深度搜索-从下向上（分治法）

```Python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        
        if root is None:
            return []
        
        left_result = self.preorderTraversal(root.left)
        right_result = self.preorderTraversal(root.right)
        
        return [root.val] + left_result + right_result
```

注意点：

> DFS 深度搜索（从上到下） 和分治法区别：前者一般将最终结果通过指针参数传入，后者一般递归返回结果最后合并

#### [BFS 层次遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

```Python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        
        levels = []
        if root is None:
            return levels
        
        bfs = collections.deque([root])
        
        while len(bfs) > 0:
            levels.append([])
            
            level_size = len(bfs)
            for _ in range(level_size):
                node = bfs.popleft()
                levels[-1].append(node.val)
                
                if node.left is not None:
                    bfs.append(node.left)
                if node.right is not None:
                    bfs.append(node.right)
        
        return levels
```

### 分治法应用

先分别处理局部，再合并结果

适用场景

- 快速排序
- 归并排序
- 二叉树相关问题

分治法模板

- 递归返回条件
- 分段处理
- 合并结果

## 常见题目示例

### [maximum-depth-of-binary-tree](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

> 给定一个二叉树，找出其最大深度。

- 思路 1：分治法

```Python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        
        if root is None:
            return 0
        
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
```

- 思路 2：层序遍历

```Python
class Solution:
    def maxDepth(self, root: TreeNode) -> List[List[int]]:
        
        depth = 0
        if root is None:
            return depth
        
        bfs = collections.deque([root])
        
        while len(bfs) > 0:
            depth += 1
            level_size = len(bfs)
            for _ in range(level_size):
                node = bfs.popleft()
                if node.left is not None:
                    bfs.append(node.left)
                if node.right is not None:
                    bfs.append(node.right)
        
        return depth
```

### [balanced-binary-tree](https://leetcode-cn.com/problems/balanced-binary-tree/)

> 给定一个二叉树，判断它是否是高度平衡的二叉树。

- 思路 1：分治法，左边平衡 && 右边平衡 && 左右两边高度 <= 1，

```Python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
 
        def depth(root):
            
            if root is None:
                return 0, True
            
            dl, bl = depth(root.left)
            dr, br = depth(root.right)
            
            return max(dl, dr) + 1, bl and br and abs(dl - dr) < 2
        
        _, out = depth(root)
        
        return out
```

- 思路 2：使用后序遍历实现分治法的迭代版本

```Python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:

        s = [[TreeNode(), -1, -1]]
        node, last = root, None
        while len(s) > 1 or node is not None:
            if node is not None:
                s.append([node, -1, -1])
                node = node.left
                if node is None:
                    s[-1][1] = 0
            else:
                peek = s[-1][0]
                if peek.right is not None and last != peek.right:
                    node = peek.right
                else:
                    if peek.right is None:
                        s[-1][2] = 0
                    last, dl, dr = s.pop()
                    if abs(dl - dr) > 1:
                        return False
                    d = max(dl, dr) + 1
                    if s[-1][1] == -1:
                        s[-1][1] = d
                    else:
                        s[-1][2] = d
        
        return True
```

### [binary-tree-maximum-path-sum](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

> 给定一个**非空**二叉树，返回其最大路径和。

- 思路：分治法。最大路径的可能情况：左子树的最大路径，右子树的最大路径，或通过根结点的最大路径。其中通过根结点的最大路径值等于以左子树根结点为端点的最大路径值加以右子树根结点为端点的最大路径值再加上根结点值，这里还要考虑有负值的情况即负值路径需要丢弃不取。

```Python
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        
        self.maxPath = float('-inf')
        
        def largest_path_ends_at(node):
            if node is None:
                return float('-inf')
            
            e_l = largest_path_ends_at(node.left)
            e_r = largest_path_ends_at(node.right)
            
            self.maxPath = max(self.maxPath, node.val + max(0, e_l) + max(0, e_r), e_l, e_r)
            
            return node.val + max(e_l, e_r, 0)
        
        largest_path_ends_at(root)
        return self.maxPath
```

### [lowest-common-ancestor-of-a-binary-tree](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

> 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

- 思路：分治法，有左子树的公共祖先或者有右子树的公共祖先，就返回子树的祖先，否则返回根节点

```Python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        
        if root is None:
            return None
        
        if root == p or root == q:
            return root
        
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        
        if left is not None and right is not None:
            return root
        elif left is not None:
            return left
        elif right is not None:
            return right
        else:
            return None
```

### BFS 层次应用

### [binary-tree-zigzag-level-order-traversal](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)

> 给定一个二叉树，返回其节点值的锯齿形层次遍历。Z 字形遍历

- 思路：在BFS迭代模板上改用双端队列控制输出顺序

```Python
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        
        levels = []
        if root is None:
            return levels
        
        s = collections.deque([root])

        start_from_left = True
        while len(s) > 0:
            levels.append([])
            level_size = len(s)
            
            if start_from_left:
                for _ in range(level_size):
                    node = s.popleft()
                    levels[-1].append(node.val)
                    if node.left is not None:
                        s.append(node.left)
                    if node.right is not None:
                        s.append(node.right)
            else:
                for _ in range(level_size):
                    node = s.pop()
                    levels[-1].append(node.val)
                    if node.right is not None:
                        s.appendleft(node.right)
                    if node.left is not None:
                        s.appendleft(node.left)
            
            start_from_left = not start_from_left
            
        
        return levels
```

### 二叉搜索树应用

### [validate-binary-search-tree](https://leetcode-cn.com/problems/validate-binary-search-tree/)

> 给定一个二叉树，判断其是否是一个有效的二叉搜索树。

- 思路 1：中序遍历后检查输出是否有序，缺点是如果不平衡无法提前返回结果， 代码略

- 思路 2：分治法，一个二叉树为合法的二叉搜索树当且仅当左右子树为合法二叉搜索树且根结点值大于右子树最小值小于左子树最大值。缺点是若不用迭代形式实现则无法提前返回，而迭代实现右比较复杂。

```Python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        
        if root is None: return True
        
        def valid_min_max(node):
            
            isValid = True
            if node.left is not None:
                l_isValid, l_min, l_max = valid_min_max(node.left)
                isValid = isValid and node.val > l_max
            else:
                l_isValid, l_min = True, node.val

            if node.right is not None:
                r_isValid, r_min, r_max = valid_min_max(node.right)
                isValid = isValid and node.val < r_min
            else:
                r_isValid, r_max = True, node.val

                
            return l_isValid and r_isValid and isValid, l_min, r_max
        
        return valid_min_max(root)[0]
```

- 思路 3：利用二叉搜索树的性质，根结点为左子树的右边界，右子树的左边界，使用先序遍历自顶向下更新左右子树的边界并检查是否合法，迭代版本实现简单且可以提前返回结果。

```Python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        
        if root is None:
            return True
        
        s = [(root, float('-inf'), float('inf'))]
        while len(s) > 0:
            node, low, up = s.pop()
            if node.left is not None:
                if node.left.val <= low or node.left.val >= node.val:
                    return False
                s.append((node.left, low, node.val))
            if node.right is not None:
                if node.right.val <= node.val or node.right.val >= up:
                    return False
                s.append((node.right, node.val, up))
        return True
```

#### [insert-into-a-binary-search-tree](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/)

> 给定二叉搜索树（BST）的根节点和要插入树中的值，将值插入二叉搜索树。 返回插入后二叉搜索树的根节点。

- 思路：如果只是为了完成任务则找到最后一个叶子节点满足插入条件即可。但此题深挖可以涉及到如何插入并维持平衡二叉搜索树的问题，并不适合初学者。

```Python
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        
        if root is None:
            return TreeNode(val)
        
        node = root
        while True:
            if val > node.val:
                if node.right is None:
                    node.right = TreeNode(val)
                    return root
                else:
                    node = node.right
            else:
                if node.left is None:
                    node.left = TreeNode(val)
                    return root
                else:
                    node = node.left
```

## 总结

- 掌握二叉树递归与非递归遍历
- 理解 DFS 前序遍历与分治法
- 理解 BFS 层次遍历

## 练习

- [ ] [maximum-depth-of-binary-tree](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)
- [ ] [balanced-binary-tree](https://leetcode-cn.com/problems/balanced-binary-tree/)
- [ ] [binary-tree-maximum-path-sum](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)
- [ ] [lowest-common-ancestor-of-a-binary-tree](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)
- [ ] [binary-tree-level-order-traversal](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)
- [ ] [binary-tree-level-order-traversal-ii](https://leetcode-cn.com/problems/binary-tree-level-order-traversal-ii/)
- [ ] [binary-tree-zigzag-level-order-traversal](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)
- [ ] [validate-binary-search-tree](https://leetcode-cn.com/problems/validate-binary-search-tree/)
- [ ] [insert-into-a-binary-search-tree](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/)
