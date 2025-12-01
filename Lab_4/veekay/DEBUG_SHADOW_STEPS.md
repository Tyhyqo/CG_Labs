# Отладка Shadow Mapping

## Проблема
Тени не отображаются на сцене, хотя:
- ✅ Код компилируется без ошибок
- ✅ Нет ошибок валидации Vulkan
- ✅ Warning исправлен

## Возможные причины

### 1. Shadow map не заполняется данными
**Проверка**: В shadow pass рендерим ли мы модели?
- Проверить, что models.size() > 0
- Проверить, что shadow pipeline корректно настроен

### 2. Light space matrix неправильная
**Проверка**: Матрица может быть вырожденной или неправильно ориентированной
- В `calculateLightSpaceMatrix()` ортогональная проекция покрывает область [-10, 10]
- Позиция "камеры света" на расстоянии 20 единиц от центра

### 3. Направление света
**Проверка**: Если свет направлен не на сцену, тени не будут видны
- По умолчанию: `dir_direction = {-0.3f, -1.0f, -0.5f}`

### 4. Bias слишком большой
**Проверка**: В shader.frag `bias = 0.005` - может скрывать тени

## Шаги отладки

### Шаг 1: Визуализация shadow value
Временно измените shader.frag, строку с `result`:
```glsl
// Вместо:
vec3 result = dir_ambient + shadow * (dir_diffuse + dir_specular);

// Используйте:
vec3 result = vec3(shadow); // Покажет тени как grayscale
```

Пересоберите шейдеры командой:
```powershell
cd C:\Education\CG_Labs\Lab_4\veekay\shaders
glslc shader.frag -o shader.frag.spv
```

**Ожидаемый результат**: 
- Белые области = нет тени (shadow = 1.0)
- Черные области = полная тень (shadow = 0.0)

Если вся сцена белая - тени не вычисляются или shadow map пустая.

### Шаг 2: Проверка frag_position_light_space
Добавьте отладочный вывод в shader.frag:
```glsl
void main() {
    // ... существующий код ...
    
    // DEBUG: Визуализация light space координат
    vec3 proj_coords = frag_position_light_space.xyz / frag_position_light_space.w;
    proj_coords = proj_coords * 0.5 + 0.5;
    out_color = vec4(proj_coords, 1.0);
    return; // Ранний выход для отладки
}
```

**Ожидаемый результат**: RGB цвет показывает (X, Y, depth) в light space
- Если черное - координаты отрицательные или нулевые (плохо)
- Если разноцветное - координаты правильные (хорошо)

### Шаг 3: Проверка модели в shadow pass
Добавьте printf debug или логирование в main.cpp после shadow pass:
```cpp
// После vkCmdEndRenderPass(cmd); в shadow pass
if (models.size() == 0) {
    std::cerr << "WARNING: No models to render in shadow pass!\n";
}
```

### Шаг 4: Проверить light_space_matrix
Добавьте вывод матрицы в update():
```cpp
veekay::mat4 light_space_matrix = calculateLightSpaceMatrix(light_dir);

// DEBUG: Вывести первую строку матрицы
std::cout << "Light space matrix[0]: " 
          << light_space_matrix[0] << " "
          << light_space_matrix[1] << " "
          << light_space_matrix[2] << " "
          << light_space_matrix[3] << std::endl;
```

**Ожидаемый результат**: Числа не должны быть NaN или Inf

## Быстрое решение

Попробуйте увеличить область ортогональной проекции в `calculateLightSpaceMatrix`:
```cpp
// Измените строки ~520-530
float ortho_size = 20.0f; // было 10.0f
```

Или уменьшите bias в shader.frag:
```glsl
float bias = 0.001; // было 0.005
```
