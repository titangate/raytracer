from geometry import BoundingBox, GeometryObject


class BoundingBoxes(object):
    def __init__(self, boxes):
        self.x_boxes = sorted(boxes, key=lambda box:box.get_mid_x())
        self.y_boxes = sorted(boxes, key=lambda box:box.get_mid_y())
        self.z_boxes = sorted(boxes, key=lambda box:box.get_mid_z())

        self.min_x = min(b.x0 for b in boxes)
        self.max_x = max(b.x1 for b in boxes)

        self.min_y = min(b.y0 for b in boxes)
        self.max_y = max(b.y1 for b in boxes)

        self.min_z = min(b.z0 for b in boxes)
        self.max_z = max(b.z1 for b in boxes)

    def get_bounding_box(self):
        return BoundingBox(self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z, None)


class KDTree(GeometryObject):
    def __init__(self, bounding_boxes, leaf=False):
        super(KDTree, self).__init__()
        self.left = None
        self.right = None
        self.leaf_items = None
        if leaf:
            self.leaf_items = bounding_boxes.x_boxes
        else:
            self.sub_divide(bounding_boxes)
        self.bounding_box = bounding_boxes.get_bounding_box()

    def find_division_point(self, bounding_boxes):
        max_diff = bounding_boxes.max_x - bounding_boxes.min_x
        axis = 'x'
        if bounding_boxes.max_y - bounding_boxes.min_y > max_diff:
            max_diff = bounding_boxes.max_y - bounding_boxes.max_y
            axis = 'y'
        if bounding_boxes.max_z - bounding_boxes.min_z > max_diff:
            axis = 'z'
        # right now naively return the center idx
        print 'dividing on %s, with %d items' % (axis, len(bounding_boxes.x_boxes))
        return axis, len(bounding_boxes.x_boxes) / 2

    def sub_divide(self, bounding_boxes):
        if len(bounding_boxes.x_boxes) <= 4:
            self.leaf_items = bounding_boxes.x_boxes
        else:
            left = []
            right = []
            axis, idx = self.find_division_point(bounding_boxes)
            if axis == 'x':
                center_x = bounding_boxes.x_boxes[idx].x0
                for box in bounding_boxes.x_boxes:
                    if box.x1 > center_x:
                        right.append(box)
                    if box.x0 <= center_x:
                        left.append(box)
            elif axis == 'y':
                center_y = bounding_boxes.y_boxes[idx].y0
                for box in bounding_boxes.y_boxes:
                    if box.y1 > center_y:
                        right.append(box)
                    if box.y0 <= center_y:
                        left.append(box)
            elif axis == 'z':
                center_z = bounding_boxes.z_boxes[idx].z0
                for box in bounding_boxes.z_boxes:
                    if box.z1 > center_z:
                        right.append(box)
                    if box.z0 <= center_z:
                        left.append(box)

            print len(left), len(right), len(bounding_boxes.x_boxes)

            if not left or not right:
                self.leaf_items = bounding_boxes.x_boxes
            elif len(set(left).intersection(right)) > len(bounding_boxes.x_boxes) / 2:
                self.leaf_items = bounding_boxes.x_boxes
            else:
                if len(left) == len(bounding_boxes.x_boxes):
                    self.left = KDTree(BoundingBoxes(left), leaf=True)
                else:
                    self.left = KDTree(BoundingBoxes(left))
                if len(right) == len(bounding_boxes.x_boxes):
                    self.right = KDTree(BoundingBoxes(right), leaf=True)
                else:
                    self.right = KDTree(BoundingBoxes(right))

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

    def shadow_hit(self, ray):
        objs = self.find_all_potential_colliders(ray)

        for obj in objs:
            if obj.shadow_hit(ray):
                return True

        return False

    def hit(self, ray):
        objs = self.find_all_potential_colliders(ray)

        sr = None
        for obj in objs:
            n_sr = obj.hit(ray)
            if sr is None or (n_sr is not None and sr.tmin > n_sr.tmin):
                sr = n_sr

        return sr
