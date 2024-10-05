from pdf2image import convert_from_path
import os

def pdf_to_png(pdf_path, output_folder):
    # Создаем папку для вывода, если ее нет
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Конвертируем PDF в список изображений (страниц)
    images = convert_from_path(pdf_path)

    for i, image in enumerate(images):
        # Сохраняем каждую страницу как PNG файл
        output_path = os.path.join(output_folder, f'page_{i + 1}.png')
        image.save(output_path, 'PNG')

    print(f"Все страницы сохранены в папку {output_folder}")

# Пример использования
pdf_path = 'artifacts/residual_ae-b_2-lr_3e04-b_s_2/epoch_999/residual_ae-b_2-lr_3e04-b_s_2.pdf'  # Укажите путь к вашему PDF файлу
output_folder = 'artifacts/residual_ae-b_2-lr_3e04-b_s_2/epoch_999/'  # Папка для сохранения PNG изображений
pdf_to_png(pdf_path, output_folder)
