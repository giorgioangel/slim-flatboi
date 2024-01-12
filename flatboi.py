import igl
import os
import argparse
import numpy as np
import open3d as o3d
from PIL import Image
from math import sqrt
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

def print_array_to_file(array, file_path):
    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Write each element of the array to the file
        for element in array:
            file.write(str(element) + '\n')

class Flatboi:
    input_obj: str
    max_iter: int
    def __init__(self, obj_path: str, max_iter: int):
        self.input_obj = obj_path
        self.max_iter = max_iter
        self.read_mesh()

    def read_mesh(self):
        self.mesh = o3d.io.read_triangle_mesh(self.input_obj)
        self.vertices = np.asarray(self.mesh.vertices, dtype=np.float64)
        self.triangles = np.asarray(self.mesh.triangles, dtype=np.int32)
        self.abf_uvs = np.asarray(self.mesh.triangle_uvs, dtype=np.float64)
    def generate_boundary(self):
        return igl.boundary_loop(self.triangles)
    
    def harmonic_ic(self):
        bnd = self.generate_boundary()
        bnd_uv = igl.map_vertices_to_circle(self.vertices, bnd)
        uv = igl.harmonic(self.vertices, self.triangles, bnd, bnd_uv, 1)
        return bnd, bnd_uv, uv
    
    def original_ic(self):
        input_directory = os.path.dirname(self.input_obj)
        base_file_name, _ = os.path.splitext(os.path.basename(self.input_obj))
        tif_path = os.path.join(input_directory, f"{base_file_name}.tif")

        # Check if the .mtl file exists
        if not os.path.exists(tif_path):
            raise FileNotFoundError("No .tif file found.")
        
        with Image.open(tif_path) as img:
            width, height = img.size
        
        uv = np.zeros((self.vertices.shape[0], 2), dtype=np.float64)
        uvs = self.abf_uvs.reshape((self.triangles.shape[0], self.triangles.shape[1], 2))
        for t in range(self.triangles.shape[0]):
            for v in range(self.triangles.shape[1]):
                uv[self.triangles[t,v]] = uvs[t,v]

        # Multiply uv coordinates by image dimensions
        uv[:, 0] *= width
        uv[:, 1] *= height

        bnd = self.generate_boundary()
        bnd_uv = np.zeros((bnd.shape[0], 2), dtype=np.float64)

        for i in range(bnd.shape[0]):
            bnd_uv[i] = uv[bnd[i]]

        return bnd, bnd_uv, uv
    
    def slim(self, initial_condition='original'):
        if initial_condition == 'original':
            bnd, bnd_uv, uv = self.original_ic()
            l2, linf, area_error = self.stretch_metrics(uv)
            print(f"Stretch metrics ABF L2: {l2:.5f}, Linf: {linf:.5f}, Area Error: {area_error:.5f}", end="\n")
        elif initial_condition == 'harmonic':
            bnd, bnd_uv, uv = self.harmonic_ic()

        slim = igl.SLIM(self.vertices, self.triangles, v_init=uv, b=bnd, bc=bnd_uv, energy_type=igl.SLIM_ENERGY_TYPE_SYMMETRIC_DIRICHLET, soft_penalty=0)

        energies = np.zeros(self.max_iter+1, dtype=np.float64)
        energies[0] = slim.energy()
        for i in tqdm(range(self.max_iter)):
            slim.solve(1)
            energies[i+1] = slim.energy()

        l2, linf, area_error = self.stretch_metrics(slim.vertices())
        print(f"Stretch metrics SLIM from {initial_condition} L2: {l2:.5f}, Linf: {linf:.5f}, , Area Error: {area_error:.5f}", end="\n")
        return slim.vertices(), energies
    
    @staticmethod
    def normalize(uv):
        uv_min = np.min(uv, axis=0)
        uv_max = np.max(uv, axis=0)
        new_uv = (uv - uv_min) / (uv_max - uv_min)
        return new_uv
    
    def save_img(self, uv):
        input_directory = os.path.dirname(self.input_obj)
        base_file_name, _ = os.path.splitext(os.path.basename(self.input_obj))
        image_path = os.path.join(input_directory, f"{base_file_name}_flatboi.png")
        min_x, min_y = np.min(uv, axis=0)
        shifted_coords = uv - np.array([min_x, min_y])
        max_x, max_y = np.max(shifted_coords, axis=0)
        # Create a white image of the determined size
        image_size = (int(round(max_y)) + 1, int(round(max_x)) + 1)
        white_image = np.ones((image_size[0], image_size[1]), dtype=np.uint16) * 65535

        # Save the grayscale image
        Image.fromarray(white_image, mode='L').save(image_path)

    def save_mtl(self):
        input_directory = os.path.dirname(self.input_obj)
        base_file_name, _ = os.path.splitext(os.path.basename(self.input_obj))
        mtl_path = os.path.join(input_directory, f"{base_file_name}.mtl")

        # Check if the .mtl file exists
        if not os.path.exists(mtl_path):
            raise FileNotFoundError("No .mtl file found.")
        
        new_file_path = os.path.join(input_directory, f"{base_file_name}_flatboi.mtl")

        # Read the contents of the original file
        with open(mtl_path, 'r') as file:
            lines = file.readlines()

        # Process each line
        for i, line in enumerate(lines):
            if line.startswith('map_Kd'):
                # Split the line by space and get the filename
                parts = line.split()
                if len(parts) > 1:
                    # Extract the base filename without extension
                    base_name = parts[1].split('.')[0]
                    # Replace the line with the new filename
                    lines[i] = f"map_Kd {base_name}_flatboi.png\n"

        # Write the updated contents to the new file
        with open(new_file_path, 'w') as file:
            file.writelines(lines)



    def save_obj(self, uv):
        input_directory = os.path.dirname(self.input_obj)
        base_file_name, _ = os.path.splitext(os.path.basename(self.input_obj))
        obj_path = os.path.join(input_directory, f"{base_file_name}_flatboi.obj")
        normalized_uv = self.normalize(uv)
        slim_uvs = np.zeros((self.triangles.shape[0],3,2), dtype=np.float64)
        for t in range(self.triangles.shape[0]):
            for v in range(self.triangles.shape[1]):
                slim_uvs[t,v,:] = normalized_uv[self.triangles[t,v]]
        slim_uvs = slim_uvs.reshape(-1,2)
        self.mesh.triangle_uvs = o3d.utility.Vector2dVector(slim_uvs)
        o3d.io.write_triangle_mesh(obj_path, self.mesh)

    def stretch_triangle(self, triangle_3d, triangle_2d):
        q1, q2, q3 = triangle_3d

        s1, t1 = triangle_2d[0]
        s2, t2 = triangle_2d[1]
        s3, t3 = triangle_2d[2]

        A = ((s2-s1)*(t3-t1)-(s3-s1)*(t2-t1))/2 # 2d area
        Ss = (q1*(t2-t3)+q2*(t3-t1)+q3*(t1-t2))/(2*A)
        St = (q1*(s3-s2)+q2*(s1-s3)+q3*(s2-s1))/(2*A)
        a = np.dot(Ss,Ss)
        b = np.dot(Ss,St)
        c = np.dot(St,St)

        G = sqrt(((a+c)+sqrt((a-c)**2+4*b**2))/2)

        L2 = sqrt((a+c)/2)

        ab = np.linalg.norm(q2-q1)
        bc = np.linalg.norm(q3-q2)
        ca = np.linalg.norm(q1-q3)
        s = (ab+bc+ca)/2
        area = sqrt(s*(s-ab)*(s-bc)*(s-ca)) # 3d area

        
        return L2, G, area, A
    
    def stretch_metrics(self, uv):
        if len(uv.shape) == 2:
            temp = uv.copy()
            uv = np.zeros((self.triangles.shape[0],3,2), dtype=np.float64)
            for t in range(self.triangles.shape[0]):
                for v in range(self.triangles.shape[1]):
                    uv[t,v,:] = temp[self.triangles[t,v]]

        linf_all = np.zeros(self.triangles.shape[0])
        area_all = np.zeros(self.triangles.shape[0])
        area2d_all = np.zeros(self.triangles.shape[0])
        per_triangle_area = np.zeros(self.triangles.shape[0])

        nominator = 0
        for t in range(self.triangles.shape[0]):
            t3d = [self.vertices[self.triangles[t,i]] for i in range(self.triangles.shape[1])]
            t2d = [uv[t,i] for i in range(self.triangles.shape[1])]

            l2, linf, area, area2d = self.stretch_triangle(t3d, t2d)

            linf_all[t] = linf
            area_all[t] = area
            area2d_all[t] = area2d
            nominator += (l2**2)*area_all[t]
            
        l2_mesh = sqrt( nominator / np.sum(area_all))
        linf_mesh = np.max(linf_all)

        alpha = area_all/np.sum(area_all)
        beta = area2d_all/np.sum(area2d_all)

        for t in range(self.triangles.shape[0]):
            if alpha[t] > beta[t]:
                per_triangle_area[t] = 1 - beta[t]/alpha[t]
            else:
                per_triangle_area[t] = 1 - alpha[t]/beta[t]
        
        return l2_mesh, linf_mesh, np.mean(per_triangle_area)
        





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reflatten .obj")
    parser.add_argument('input', type=str, help='Path to .obj to reflatten.')
    parser.add_argument('iter', type=int, help='Max number of iterations.')

    args = parser.parse_args()

    # Check if the input file exists and is a .obj file
    if not os.path.exists(args.input):
        print(f"Error: The file '{args.input}' does not exist.")
        exit(1)

    if not args.input.lower().endswith('.obj'):
        print(f"Error: The file '{args.input}' is not a .obj file.")
        exit(1)

    assert args.iter > 0, "Max number of iterations should be positive."

    # Get the directory of the input file
    input_directory = os.path.dirname(args.input)
    # Filename for the energies file
    energies_file = os.path.join(input_directory, 'energies_flatboi.txt')

    try:
        flatboi = Flatboi(args.input, args.iter)

        # trying different initial conditions
        original_uvs, original_energies = flatboi.slim(initial_condition='original')
        harmonic_uvs, harmonic_energies = flatboi.slim(initial_condition='harmonic')
        
        if original_energies[-1] < harmonic_energies[-1]:
            print(f"Selected ABF initial condition")
            flatboi.save_img(original_uvs)
            flatboi.save_obj(original_uvs)
            print_array_to_file(original_energies, energies_file)
        else:
            print(f"Selected harmonic initial condition")
            flatboi.save_img(harmonic_uvs)
            flatboi.save_obj(harmonic_uvs)
            print_array_to_file(harmonic_energies, energies_file)
        
        flatboi.save_mtl()

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)


