import gradio as gr
import os
import json
from core import analyze_image, save_passport
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from io import BytesIO

image_folder = "input"
passports_folder = "passports"
photo_cache = {}
meta_cache = {}
entity_index = {}

# Загрузка всех изображений и их анализ-статус

def load_gallery(filter_type=None, date_range=None, score_range=None):
    images = []
    meta_cache.clear()
    entity_index.clear()
    for file in sorted(os.listdir(image_folder)):
        if file.lower().endswith(('.jpg', '.png')):
            filepath = os.path.join(image_folder, file)
            thumb = Image.open(filepath).resize((150, 150))
            json_path = os.path.join(passports_folder, file.replace('.jpg', '.json').replace('.png', '.json'))
            analyzed = os.path.exists(json_path)
            mask_score = None
            type_flag = "⚪"
            date_val = ""
            entity_id = ""
            anomaly = ""
            if analyzed:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    mask_score = data.get("mask_score")
                    date_val = data.get("date", "")
                    entity_id = data.get("entity_id", "")
                    anomaly = "🚨" if data.get("anomaly_flag") else ""
                    if mask_score is not None:
                        if mask_score > 0.75:
                            type_flag = "🔴"
                        elif mask_score > 0.4:
                            type_flag = "🟡"
                        else:
                            type_flag = "🟢"
                    meta_cache[file] = {
                        "score": mask_score,
                        "type": type_flag,
                        "date": date_val,
                        "entity": entity_id,
                        "anomaly": anomaly
                    }
                    if entity_id:
                        entity_index.setdefault(entity_id, []).append((date_val, file))
            if filter_type and filter_type != "Все" and type_flag != filter_type:
                continue
            if date_range and date_val:
                try:
                    d = int(date_val.replace('.', ''))
                    if d < date_range[0] or d > date_range[1]:
                        continue
                except:
                    continue
            if score_range and mask_score is not None:
                if mask_score < score_range[0] or mask_score > score_range[1]:
                    continue
            status = f"{type_flag} {file}  {anomaly} ({entity_id})"
            images.append((thumb, status))
            photo_cache[file] = filepath
    return images

def view_analysis(filename):
    filepath = photo_cache.get(filename)
    if not filepath:
        return None, "Файл не найден."

    result = analyze_image(filepath)
    if result:
        save_passport(result)
        info = f"Дата: {result['date']}\nMask score: {result['mask_score']}\nПричины: {', '.join(result['mask_score_reason'])}\nСущность: {result['entity_id']}\nАномалия: {'Да' if result.get('anomaly_flag') else 'Нет'}\nИнтервал: {result.get('delta_days', '-')} дней"
        image = Image.open(filepath).resize((512, 512))
        return image, info
    else:
        return None, "Ошибка анализа."

def view_passport_raw(filename):
    json_path = os.path.join(passports_folder, filename.replace('.jpg', '.json').replace('.png', '.json'))
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return f.read()
    return "JSON не найден."

def get_entity_profile(entity_id):
    if entity_id not in entity_index:
        return "Нет данных.", None
    records = []
    for date_val, fname in sorted(entity_index[entity_id]):
        path = os.path.join(passports_folder, fname.replace('.jpg', '.json').replace('.png', '.json'))
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                score = data.get("mask_score")
                anomaly = "🚨" if data.get("anomaly_flag") else ""
                records.append((date_val, score, anomaly))
    # Текстовая история
    text_lines = [f"📂 Сущность {entity_id}"] + [f"{d}: score {s} {a}" for d, s, a in records]
    # График
    if not records:
        return "Нет данных.", None
    dates = [r[0] for r in records]
    scores = [r[1] for r in records]
    anomalies = [i for i, r in enumerate(records) if r[2] == "🚨"]
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(scores, marker='o', label='mask_score')
    for i in anomalies:
        ax.plot(i, scores[i], 'ro')
    ax.set_title(f"Сущность {entity_id} — mask_score во времени")
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=45, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True)
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return "\n".join(text_lines), buf

def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## 🧠 Галерея лиц: выбор изображения для анализа")

        with gr.Row():
            filter_type = gr.Radio(["Все", "🟢", "🟡", "🔴"], label="Фильтр по типу", value="Все")
            date_range = gr.Slider(20000101, 20991231, step=1, label="Дата (числовой диапазон)", value=(20000101, 20991231))
            score_range = gr.Slider(0.0, 1.0, step=0.01, label="Диапазон mask_score", value=(0.0, 1.0))

        gallery = gr.Gallery(label="Все фото", columns=6, height=600)
        selected = gr.Textbox(label="Имя файла")
        analyze_btn = gr.Button("🔍 Проанализировать это фото")
        output_image = gr.Image(label="Результат анализа", type="pil")
        output_text = gr.Textbox(label="Основные метрики", lines=6)
        view_json_btn = gr.Button("📄 Показать JSON-паспорт")
        json_box = gr.Textbox(label="JSON (паспорт изображения)", lines=15)
        profile_btn = gr.Button("👥 Показать профиль сущности")
        entity_box = gr.Textbox(label="🧬 История по сущности", lines=12)
        entity_chart = gr.Image(label="📈 График mask_score по сущности")

        def update_gallery_view(t, d, s):
            return load_gallery(t, d, s)

        def on_select(evt):
            selected.update(value=evt.index[1])

        def run_analysis(name):
            return view_analysis(name)

        def show_json(name):
            return view_passport_raw(name)

        def show_entity(name):
            parts = name.split()
            for part in parts:
                if part.startswith("entity_"):
                    return get_entity_profile(part)
            return "Сущность не определена.", None

        filter_type.change(fn=update_gallery_view, inputs=[filter_type, date_range, score_range], outputs=[gallery])
        date_range.change(fn=update_gallery_view, inputs=[filter_type, date_range, score_range], outputs=[gallery])
        score_range.change(fn=update_gallery_view, inputs=[filter_type, date_range, score_range], outputs=[gallery])
        gallery.select(fn=on_select, inputs=[], outputs=[selected])
        analyze_btn.click(fn=run_analysis, inputs=[selected], outputs=[output_image, output_text])
        view_json_btn.click(fn=show_json, inputs=[selected], outputs=[json_box])
        profile_btn.click(fn=show_entity, inputs=[selected], outputs=[entity_box, entity_chart])

        gr.Markdown("---")
        gr.Markdown("### ⚙️ Инструкции")
        gr.Markdown("1. Выберите фильтры по типу, дате или mask_score.\n2. Нажмите на фото.\n3. Запустите анализ.\n4. Посмотрите mask_score и JSON.\n5. Нажмите 👥 чтобы увидеть хронологию сущности и график изменений.")

    return demo

if __name__ == "__main__":
    ui = build_interface()
    ui.launch()
