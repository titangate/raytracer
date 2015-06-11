from geometry import BoundingBox, GeometryObject
import sys

INF = sys.maxint
epsilon = 1.0e-7


class BoundingBoxes(object):
    def __init__(self, boxes):
        self.x_boxes = sorted(boxes, key=lambda box:box.get_mid_x())
        self.y_boxes = sorted(boxes, key=lambda box:box.get_mid_y())
        self.z_boxes = sorted(boxes, key=lambda box:box.get_mid_z())

        self.x0 = min(b.x0 for b in boxes)
        self.x1 = max(b.x1 for b in boxes)

        self.y0 = min(b.y0 for b in boxes)
        self.y1 = max(b.y1 for b in boxes)

        self.z0 = min(b.z0 for b in boxes)
        self.z1 = max(b.z1 for b in boxes)

    def get_bounding_box(self):
        return BoundingBox(self.x0, self.x1, self.y0, self.y1, self.z0, self.z1, None)


def SAH(array, axis, min_d, max_d):
    array.sort(key=lambda box:box.data[axis][1])
    best_cost = INF
    best_left = None
    best_right = None
    split_center = None
    x0_idx = 0
    x0_sorted_array = sorted(array, key=lambda box:box.data[axis][0])
    for i, box in enumerate(array):
        plane = box.data[axis][1] + epsilon
        split_idx = i
        while split_idx < len(array) and array[split_idx].data[axis][1] <= plane:
            split_idx += 1

        while x0_idx < len(array) and x0_sorted_array[x0_idx].data[axis][0] <= plane:
            x0_idx += 1
        cost = (x0_idx + 1) * (plane - min_d) + (len(array) - split_idx - 1) * (max_d - plane)
        if cost < best_cost:
            best_cost = cost
            best_left = x0_idx
            best_right = split_idx
            split_center = plane

    split_center = min(max_d, max(split_center, min_d))

    return best_cost, x0_sorted_array[:best_left], array[best_right:], split_center


class KDTree(GeometryObject):
    def __init__(self, bounding_boxes, leaf=False, min_d=None, max_d=None):
        super(KDTree, self).__init__()
        self.left = None
        self.right = None
        self.leaf_items = None
        self.bounding_box = bounding_boxes.get_bounding_box()
        if leaf:
            self.leaf_items = bounding_boxes.x_boxes
        else:
            if min_d is None:
                min_d = [self.bounding_box.x0, self.bounding_box.y0, self.bounding_box.z0]
            if max_d is None:
                max_d = [self.bounding_box.x1, self.bounding_box.y1, self.bounding_box.z1]
            self.sub_divide(bounding_boxes, min_d, max_d)

    def find_division_point(self, bounding_boxes):
        max_diff = bounding_boxes.x1 - bounding_boxes.x0
        axis = 'x'
        if bounding_boxes.z1 - bounding_boxes.z0 > max_diff:
            max_diff = bounding_boxes.z1 - bounding_boxes.z0
            axis = 'z'
        if bounding_boxes.y1 - bounding_boxes.y0 > max_diff:
            axis = 'y'
        # right now naively return the center idx
        return axis, len(bounding_boxes.x_boxes) / 2

    def print_tree(self, depth=0):
        s = '\t' * depth
        if self.leaf_items:
            s += ' '.join(t.obj.label for t in self.leaf_items)
            print s
        else:
            print s + 'split_axis: %d' % self.split_idx
            self.left.print_tree(depth + 1)
            self.right.print_tree(depth + 1)

    def sub_divide(self, bounding_boxes, min_d, max_d):
        print len(bounding_boxes.x_boxes)
        if len(bounding_boxes.x_boxes) <= 20:
            self.leaf_items = bounding_boxes.x_boxes
        else:
            best_cost = INF
            left = None
            right = None
            for axis in xrange(3):
                cost, cur_left, cur_right, split_center = (
                    SAH(bounding_boxes.x_boxes, axis, min_d[axis], max_d[axis])
                )
                dim_a = max_d[(axis + 1) % 3] - min_d[(axis + 1) % 3]
                dim_b = max_d[(axis + 2) % 3] - min_d[(axis + 2) % 3]
                cost *= dim_a * dim_b
                if cost < best_cost:
                    best_cost = cost
                    self.split_idx = axis
                    self.split_center = split_center
                    left = cur_left
                    right = cur_right

            if not left or not right:
                self.leaf_items = bounding_boxes.x_boxes
            # elif len(set(left).intersection(right)) > len(bounding_boxes.x_boxes) / 2:
            #     self.leaf_items = bounding_boxes.x_boxes
            else:
                max_d_left = max_d[:]
                max_d_left[self.split_idx] = self.split_center
                min_d_right = min_d[:]
                min_d_right[self.split_idx] = self.split_center

                if len(left) == len(bounding_boxes.x_boxes):
                    self.left = KDTree(BoundingBoxes(left), leaf=True, max_d=max_d_left, min_d=min_d)
                else:
                    self.left = KDTree(BoundingBoxes(left), max_d=max_d_left, min_d=min_d)
                if len(right) == len(bounding_boxes.x_boxes):
                    self.right = KDTree(BoundingBoxes(right), leaf=True, max_d=max_d, min_d=min_d_right)
                else:
                    self.right = KDTree(BoundingBoxes(right), max_d=max_d, min_d=min_d_right)

    def iterate_nodes(self, ray):
        b_hit, close_distance, far_distance = self.bounding_box.hit(ray)
        stack = [(self, close_distance, far_distance)]
        if not b_hit:
            return
        while stack:
            node, close_distance, far_distance = stack.pop()
            while not node.leaf_items:
                t = (node.split_center - ray.origin[node.split_idx]) / ray.direction[node.split_idx]
                if node.split_center >= ray.origin[node.split_idx]:
                    near, far = node.left, node.right
                else:
                    near, far = node.right, node.left

                if t >= far_distance or t < 0:
                    node = near
                elif t <= close_distance:
                    node = far
                else:
                    stack.append((far, t, far_distance))
                    node = near
                    far_distance = t
            yield node

    def shadow_hit(self, ray):
        for node in self.iterate_nodes(ray):
            for box in node.leaf_items:
                hit, t = box.obj.shadow_hit(ray)
                if hit:
                    return True, t

        return False, -1

    def hit(self, ray):
        sr = None
        for node in self.iterate_nodes(ray):
            for box in node.leaf_items:
                n_sr = box.obj.hit(ray)
                if n_sr and (sr is None or (n_sr is not None and sr.tmin > n_sr.tmin)):
                    sr = n_sr
            if sr:
                return sr

    def find_all_potential_colliders(self, ray):
        if not self.bounding_box.hit(ray):
            return set()
        result = set()
        if self.leaf_items:
            for box in self.leaf_items:
                if box.hit(ray):
                    result.add(box.obj)
        else:
            result = result.union(self.left.find_all_potential_colliders(ray))
            result = result.union(self.right.find_all_potential_colliders(ray))
        return result

    # def shadow_hit(self, ray):
    #     objs = self.find_all_potential_colliders(ray)

    #     for obj in objs:
    #         if obj.shadow_hit(ray):
    #             return True

    #     return False

    # def hit(self, ray):
    #     objs = self.find_all_potential_colliders(ray)
    #     print "testing %d" % len(objs)

    #     sr = None
    #     for obj in objs:
    #         n_sr = obj.hit(ray)
    #         if sr is None or (n_sr is not None and sr.tmin > n_sr.tmin):
    #             sr = n_sr

    #     return sr
