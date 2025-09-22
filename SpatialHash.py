import pygame

class SpatialHash:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.grid = {}

    def _hash(self, x, y):
        return x // self.cell_size, y // self.cell_size

    def insert(self, obj_id, team, x, y, size):
        cell = self._hash(x, y)
        data = [obj_id, team, x, y, size]


        cells_to_remove = []

        #
        for key, objects in list(self.grid.items()):
            for obj in objects:
                if obj[0] == obj_id:
                    #print(f'ID {obj_id} ALREADY PRESENT, UPDATING...')

                    if key != cell:
                        objects.remove(obj)
                        if not self.grid[key]:
                            cells_to_remove.append(key)
                    else:

                        obj[2], obj[3], obj[4] = x, y, size
                    break

        for cell_to_remove in cells_to_remove:
            del self.grid[cell_to_remove]

        if cell not in self.grid:
            self.grid[cell] = []
        if obj_id not in [obj[0] for obj in self.grid[cell]]:
            self.grid[cell].append(data)

    def remove(self, obj_id):
        cells_to_remove = []
        for cell, objects in list(self.grid.items()):
            for obj in objects:
                if obj[0] == obj_id:
                    objects.remove(obj)
                    break  #
            if not objects:
                cells_to_remove.append(cell)

        for cell in cells_to_remove:
            del self.grid[cell]

    def get_object_ids_in_cell(self, cell):
        if cell in self.grid:
            return [obj[0] for obj in self.grid[cell]]
        return []

    def query(self, rect):
        nearby = []
        x1, y1, x2, y2 = rect.left, rect.top, rect.right, rect.bottom

        start_x, start_y = self._hash(x1, y1)
        end_x, end_y = self._hash(x2, y2)

        for x in range(start_x, end_x + 1):
            for y in range(start_y, end_y + 1):
                if (x, y) in self.grid:
                    nearby.extend(self.grid[(x, y)])

        return nearby

    def clear(self):
        self.grid.clear()

    def draw(self, screen):
        grid_color = (150, 150, 150)

        for cell, objects in self.grid.items():
            cell_x, cell_y = cell
            rect = pygame.Rect(cell_x * self.cell_size, cell_y * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(screen, grid_color, rect, 1)
