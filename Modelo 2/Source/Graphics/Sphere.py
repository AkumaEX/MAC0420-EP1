import math
import numpy as np
from OpenGL import GL
from PyQt5.QtGui import QOpenGLShaderProgram, QOpenGLShader

from Source.Graphics.Actor import Actor


class Sphere(Actor):

    # initialization
    def __init__(self, renderer, **kwargs):
        """Initialize actor."""
        super(Sphere, self).__init__(renderer, **kwargs)

        self._r = kwargs.get("r")
        self._s = kwargs.get("s")
        self._vertices = None

        # create actor
        self.initialize()

    def addVertex(self, v, vertices):
        """Add a vertex into the array"""
        vn = v / np.linalg.norm(v) * self._r
        vertices += [[vn[0], vn[1], vn[2]]]
        return len(vertices) - 1

    def generateGeometry(self):
        """Generate vertices"""
        vertices = []
        indices = []

        t = (1.0 + math.sqrt(5.0)) / 2.0

        self.addVertex(np.array((-1.0, t, 0)), vertices)
        self.addVertex(np.array((1.0, t, 0)), vertices)
        self.addVertex(np.array((-1.0, -t, 0)), vertices)
        self.addVertex(np.array((1.0, -t, 0)), vertices)

        self.addVertex(np.array((0, -1.0, t)), vertices)
        self.addVertex(np.array((0, 1.0, t)), vertices)
        self.addVertex(np.array((0, -1.0, -t)), vertices)
        self.addVertex(np.array((0, 1.0, -t)), vertices)

        self.addVertex(np.array((t, 0, -1.0)), vertices)
        self.addVertex(np.array((t, 0, 1.0)), vertices)
        self.addVertex(np.array((-t, 0, -1.0)), vertices)
        self.addVertex(np.array((-t, 0, 1.0)), vertices)

        # 5 faces around point 0
        indices += [[0, 11, 5]]
        indices += [[0, 11, 5]]
        indices += [[0, 5, 1]]
        indices += [[0, 1, 7]]
        indices += [[0, 7, 10]]
        indices += [[0, 10, 11]]

        # 5 adjacent faces 
        indices += [[1, 5, 9]]
        indices += [[5, 11, 4]]
        indices += [[11, 10, 2]]
        indices += [[10, 7, 6]]
        indices += [[7, 1, 8]]

        # 5 faces around point 3
        indices += [[3, 9, 4]]
        indices += [[3, 4, 2]]
        indices += [[3, 2, 6]]
        indices += [[3, 6, 8]]
        indices += [[3, 8, 9]]

        # 5 adjacent faces 
        indices += [[4, 9, 5]]
        indices += [[2, 4, 11]]
        indices += [[6, 2, 10]]
        indices += [[8, 6, 7]]
        indices += [[9, 8, 1]]

        self._vertices = np.array(vertices, dtype=np.float32)
        self._indices = np.array(indices, dtype=np.uint32)

    def initialize(self):
        """Creates icosahedron geometry"""
        if self._vertices is None:
            self.generateGeometry()

        # create object
        self.create(self._vertices, indices=self._indices)
        self.initShader()

    def render(self):
        """Render icosahedron"""
        self._render_mode = GL.GL_PATCHES
        GL.glDrawElements(self._render_mode, self.numberOfIndices, GL.GL_UNSIGNED_INT, None)

    def initShader(self):
        self._programShader = QOpenGLShaderProgram()
        self._programShader.addShaderFromSourceCode(QOpenGLShader.Vertex, self.vertexShaderSource())
        self._programShader.addShaderFromSourceCode(QOpenGLShader.TessellationControl,
                                                    self.tessellationControlShaderSource())
        self._programShader.addShaderFromSourceCode(QOpenGLShader.TessellationEvaluation,
                                                    self.tessellationEvalShaderSource())
        self._programShader.addShaderFromSourceCode(QOpenGLShader.Fragment, self.fragmentShaderSource())
        self._programShader.link()

        self._programFlatShader = QOpenGLShaderProgram()
        self._programFlatShader.addShaderFromSourceCode(QOpenGLShader.Vertex, self.vertexShaderSource())
        self._programFlatShader.addShaderFromSourceCode(QOpenGLShader.TessellationControl,
                                                        self.tessellationControlShaderSource())
        self._programFlatShader.addShaderFromSourceCode(QOpenGLShader.TessellationEvaluation,
                                                        self.tessellationEvalFlatShaderSource())
        self._programFlatShader.addShaderFromSourceCode(QOpenGLShader.Fragment, self.fragmentFlatShaderSource())
        self._programFlatShader.link()

        self.setSolidShader(self._programShader)
        self.setSolidFlatShader(self._programFlatShader)
        self.setNoLightSolidShader(self._programShader)
        self.setWireframeShader(self._programShader)

        self._programShader.bind()
        self._programShader.setUniformValue("radius", self._r)
        self._programShader.setUniformValue("subdivisionLevel", self._s)
        self._programShader.release()

        self._programFlatShader.bind()
        self._programFlatShader.setUniformValue("radius", self._r)
        self._programFlatShader.setUniformValue("subdivisionLevel", self._s)
        self._programFlatShader.release()

    def vertexShaderSource(self):
        return """
        #version 400
        layout(location = 0) in vec3 position;

        uniform mat4 modelMatrix;
        uniform mat4 viewMatrix;
        uniform vec4 lightPosition;
        uniform vec3 lightAttenuation;

        smooth out vec3 vertexPosition;
        smooth out vec3 lightDirection;
        smooth out float attenuation;

        void main()
        {
            vec4 Position = viewMatrix * modelMatrix * vec4(position, 1.0);
            if (lightPosition.w == 0.0) {
                lightDirection = normalize(lightPosition.xyz);
                attenuation = 1.0;
            } else {
                lightDirection = normalize(lightPosition.xyz - Position.xyz);
                float distance = length(lightPosition.xyz - Position.xyz);
                attenuation = 1.0 / (lightAttenuation.x + lightAttenuation.y * distance + lightAttenuation.z * distance * distance);
            }
            vertexPosition = position;
        }
        """

    def tessellationControlShaderSource(self):
        return """
        #version 400 core
        layout(vertices = 3) out;

        smooth in vec3 vertexPosition[];
        smooth in vec3 lightDirection[];
        smooth in float attenuation[];

        uniform int subdivisionLevel;

        smooth out vec3 tcsPosition[];
        smooth patch out vec3 tcsLightDirection;
        smooth patch out float tcsAttenuation;

        void main()
        {
            tcsPosition[gl_InvocationID] = vertexPosition[gl_InvocationID];
            tcsLightDirection = lightDirection[gl_InvocationID];
            tcsAttenuation = attenuation[gl_InvocationID];

            if (gl_InvocationID == 0) {
                gl_TessLevelInner[0] = subdivisionLevel;
                gl_TessLevelOuter[0] = subdivisionLevel;
                gl_TessLevelOuter[1] = subdivisionLevel;
                gl_TessLevelOuter[2] = subdivisionLevel;            
            }
        }
        """

    def tessellationEvalShaderSource(self):
        return """
        #version 400 core
        layout(triangles, equal_spacing, ccw) in;

        smooth in vec3 tcsPosition[];
        smooth patch in vec3 tcsLightDirection;
        smooth patch in float tcsAttenuation;

        uniform mat4 projectionMatrix;
        uniform mat4 modelMatrix;
        uniform mat4 viewMatrix;
        uniform mat3 normalMatrix;

        uniform int radius;

        smooth out vec3 vertexPosition;
        smooth out vec3 vertexNormal;
        smooth out vec3 lightDirection;
        smooth out float attenuation;
        smooth out vec2 textCoord;

        #define PI 3.14159265

        void main()
        {
            vec3 p0 = gl_TessCoord.x * tcsPosition[0];
            vec3 p1 = gl_TessCoord.y * tcsPosition[1];
            vec3 p2 = gl_TessCoord.z * tcsPosition[2];

            vec3 Position = normalize(p0 + p1 + p2);
            vertexPosition = radius * Position;
            vertexNormal = vec3(viewMatrix * vec4(normalMatrix*Position, 0));
            lightDirection = tcsLightDirection;
            attenuation = tcsAttenuation;
            textCoord.x = atan(Position.y, Position.x)/(2*PI);
            textCoord.y = 1 - acos(Position.z)/PI;

            gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(vertexPosition, 1); 
        }
        """

    def fragmentShaderSource(self):
        return """
        #version 400
        struct Material {
            vec3 emission;
            vec3 ambient;
            vec3 diffuse;
            vec3 specular;    
            float shininess;
        }; 

        struct Light {
            vec3 ambient;
            vec3 diffuse;
            vec3 specular;
        };

        smooth in vec3 vertexNormal;
        smooth in vec3 vertexPosition;
        smooth in vec3 lightDirection;
        smooth in float attenuation;
        smooth in vec2 textCoord;

        uniform Material material;
        uniform Light light;

        out vec4 fragColor;

        void main()
        {    
            // ambient term
            vec3 ambient = material.ambient * light.ambient;

            // diffuse term
            vec3 N = normalize(vertexNormal);
            vec3 L = normalize(lightDirection);
            vec3 diffuse = light.diffuse * material.diffuse * max(dot(N, L), 0.0);

            // specular term
            vec3 E = normalize(-vertexPosition);
            vec3 R = normalize(-reflect(L, N)); 
            vec3 specular = light.specular * material.specular * pow(max(dot(R, E), 0.0), material.shininess);

            // final intensity
            vec3 intensity = material.emission + clamp(ambient + attenuation * (diffuse + specular), 0.0, 1.0);
            fragColor = vec4(intensity, 1.0);
        }
        """

    def tessellationEvalFlatShaderSource(self):
        return """
        #version 400 core
        layout(triangles, equal_spacing, ccw) in;

        smooth in vec3 tcsPosition[];
        smooth patch in vec3 tcsLightDirection;
        smooth patch in float tcsAttenuation;

        uniform mat4 projectionMatrix;
        uniform mat4 modelMatrix;
        uniform mat4 viewMatrix;
        uniform mat3 normalMatrix;

        uniform int radius;

        smooth out vec3 vertexPosition;
        flat out vec3 vertexNormal;
        smooth out vec3 lightDirection;
        smooth out float attenuation;
        smooth out vec2 textCoord;

        #define PI 3.14159265

        void main()
        {
            vec3 p0 = gl_TessCoord.x * tcsPosition[0];
            vec3 p1 = gl_TessCoord.y * tcsPosition[1];
            vec3 p2 = gl_TessCoord.z * tcsPosition[2];

            vec3 Position = normalize(p0 + p1 + p2);
            vertexPosition = radius * Position;
            vertexNormal = vec3(viewMatrix * vec4(normalMatrix*Position, 0));
            lightDirection = tcsLightDirection;
            attenuation = tcsAttenuation;
            textCoord.x = atan(Position.y, Position.x)/(2*PI);
            textCoord.y = 1 - acos(Position.z)/PI;

            gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(vertexPosition, 1); 
        }
        """

    def fragmentFlatShaderSource(self):
        return """
        #version 400
        struct Material {
            vec3 emission;
            vec3 ambient;
            vec3 diffuse;
            vec3 specular;    
            float shininess;
        }; 

        struct Light {
            vec3 ambient;
            vec3 diffuse;
            vec3 specular;
        };

        flat in vec3 vertexNormal;
        smooth in vec3 vertexPosition;
        smooth in vec3 lightDirection;
        smooth in float attenuation;
        smooth in vec2 textCoord;

        uniform Material material;
        uniform Light light;

        out vec4 fragColor;

        void main()
        {    
            // ambient term
            vec3 ambient = material.ambient * light.ambient;

            // diffuse term
            vec3 N = normalize(vertexNormal);
            vec3 L = normalize(lightDirection);
            vec3 diffuse = light.diffuse * material.diffuse * max(dot(N, L), 0.0);

            // specular term
            vec3 E = normalize(-vertexPosition);
            vec3 R = normalize(-reflect(L, N)); 
            vec3 specular = light.specular * material.specular * pow(max(dot(R, E), 0.0), material.shininess);

            // final intensity
            vec3 intensity = material.emission + clamp(ambient + attenuation * (diffuse + specular), 0.0, 1.0);
            fragColor = vec4(intensity, 1.0);
        }
        """
