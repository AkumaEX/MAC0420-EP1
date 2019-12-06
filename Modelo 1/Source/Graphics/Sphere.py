import numpy as np
from OpenGL import GL
from Source.Graphics.Actor import Actor


class Sphere(Actor):

    # initialization:
    def __init__(self, renderer, **kwargs):
        """Initialize actor."""
        super(Sphere, self).__init__(renderer, **kwargs)

        self._r = kwargs.get("r", 1.0)
        self._v = kwargs.get("v", 1.0)
        self._h = kwargs.get("h", 1.0)
        self._vertices = None
        self._normals = None

        # create actor
        self.initialize()

    def initialize(self):
        """Initialize actor"""
        if self._vertices is None:
            self.generateGeometry()

        # create object
        self.create(self._vertices, normals=self._normals)

    def generateGeometry(self):
        """Creates sphere geometry"""

        # angulos discretizados
        phi = np.linspace(0.0, 2.0 * np.pi, self._h)
        theta = np.linspace(0.0, 1.0 * np.pi, self._v)

        # TOPO

        # normais
        nx = np.cos(phi)
        ny = np.sin(phi)
        nz = np.cos(theta[1])

        # vertices
        x = nx * self._r * np.sin(theta[1])
        y = ny * self._r * np.sin(theta[1])
        z = nz * self._r

        vertices, normals, texture = [[0, 0, self._r]], [[0, 0, 1]], [[0, 1]]
        for i in range(self._h):
            vertices.append([x[i], y[i], z])
            normals.append([nx[i], ny[i], nz])
            texture.append([phi[i] / (2 * np.pi), 1 - theta[1] / np.pi])
        vertices_top = np.array(vertices, dtype=np.float32)
        normals_top = np.array(normals, dtype=np.float32)
        texture_top = np.array(texture, dtype=np.float32)
        self._num_vertices_top = len(vertices_top)

        # LADO
        vertices, normals, texture = [], [], []
        for j in range(1, self._v - 1):
            nzt = np.cos(theta[j])
            nzb = np.cos(theta[j + 1])
            rt = self._r * np.sin(theta[j])
            rb = self._r * np.sin(theta[j + 1])
            xt = rt * nx
            yt = rt * ny
            xb = rb * nx
            yb = rb * ny
            zt = self._r * nzt
            zb = self._r * nzb
            for i in range(0, self._h - 1):
                vertices.append([xt[i], yt[i], zt])
                normals.append([nx[i], ny[i], nzt])
                texture.append([phi[i] / (2 * np.pi), 1 - theta[j] / np.pi])

                vertices.append([xb[i], yb[i], zb])
                normals.append([nx[i], ny[i], nzb])
                texture.append([phi[i] / (2 * np.pi), 1 - theta[j + 1] / np.pi])

                vertices.append([xt[i + 1], yt[i + 1], zt])
                normals.append([nx[i + 1], ny[i + 1], nzt])
                texture.append(
                    [phi[i + 1] / (2 * np.pi), 1 - theta[j] / np.pi])

                vertices.append([xb[i], yb[i], zb])
                normals.append([nx[i], ny[i], nzb])
                texture.append([phi[i] / (2 * np.pi), 1 - theta[j + 1] / np.pi])

                vertices.append([xb[i + 1], yb[i + 1], zb])
                normals.append([nx[i + 1], ny[i + 1], nzb])
                texture.append(
                    [phi[i + 1] / (2 * np.pi), 1 - theta[j + 1] / np.pi])

                vertices.append([xt[i + 1], yt[i + 1], zt])
                normals.append([nx[i + 1], ny[i + 1], nzt])
                texture.append(
                    [phi[i + 1] / (2 * np.pi), 1 - theta[j] / np.pi])

        vertices_side = np.array(vertices, dtype=np.float32)
        normals_side = np.array(normals, dtype=np.float32)
        texture_side = np.array(texture, dtype=np.float32)
        self._num_vertices_side = len(vertices_side)

        # BASE
        nz = np.cos(theta[-2])
        x = nx * self._r * np.sin(theta[-2])
        y = ny * self._r * np.sin(theta[-2])
        z = nz * self._r
        vertices, normals, texture = [[0, 0, -self._r]], [[0, 0, -1]], [[0, 0]]
        for i in range(self._h):
            vertices.append([x[i], y[i], z])
            normals.append([nx[i], ny[i], nz])
            texture.append([phi[i] / (2 * np.pi), 1 - theta[-2] / np.pi])
        vertices_bot = np.array(vertices, dtype=np.float32)
        normals_bot = np.array(normals, dtype=np.float32)
        texture_bot = np.array(texture, dtype=np.float32)
        self._num_vertices_bot = len(vertices_bot)

        self._vertices = np.concatenate(
            (vertices_top, vertices_side, vertices_bot))
        self._normals = np.concatenate(
            (normals_top, normals_side, normals_bot))
        self._textures = np.concatenate(
            (texture_top, texture_side, texture_bot))

    def render(self):
        """Render Sphere"""
        GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, self._num_vertices_top)
        GL.glDrawArrays(GL.GL_TRIANGLES, self._num_vertices_top,
                        self._num_vertices_side)
        GL.glDrawArrays(GL.GL_TRIANGLE_FAN, self._num_vertices_top +
                        self._num_vertices_side, self._num_vertices_bot)
