#version 330 core
in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D inputTexture; // Входная текстура (матрица)
uniform float kernel[9]; // 3x3 Кернел (фильтр)

void main() {
    vec2 tex_offset = 1.0 / textureSize(inputTexture, 0); // Размер одного пикселя
    vec3 result = vec3(0.0);

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            vec2 offset = vec2(float(i), float(j)) * tex_offset;
            result += texture(inputTexture, TexCoords + offset).rgb * kernel[(i + 1) * 3 + (j + 1)];
        }
    }

    FragColor = vec4(result, 1.0);
}