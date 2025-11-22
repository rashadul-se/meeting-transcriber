"""
Meeting Transcription & Report Generator - Streamlit App
========================================================
Multi-language support, scalable for 500 concurrent users

Installation:
    pip install streamlit openai-whisper transformers torch fpdf2 pandas python-multipart

Run:
    streamlit run app.py --server.maxUploadSize 500
"""

import streamlit as st
import whisper
import os
import io
import json
from datetime import datetime
from transformers import pipeline
from fpdf import FPDF
import tempfile
import traceback
from pathlib import Path

# ============================================================
# MULTI-LANGUAGE SUPPORT
# ============================================================

TRANSLATIONS = {
    "en": {
        "title": "🎙️ Meeting Transcription & Report Generator",
        "subtitle": "AI-Powered Meeting Analysis with Outline Method",
        "language": "Language",
        "upload_audio": "Upload Audio File",
        "audio_help": "Supported formats: MP3, WAV, M4A, WebM, MP4, FLAC, OGG (Max 500MB)",
        "meeting_details": "Meeting Details (Optional)",
        "meeting_title": "Meeting Title",
        "meeting_date": "Meeting Date",
        "meeting_time": "Meeting Time",
        "location": "Location",
        "organizer": "Organizer",
        "attendees": "Attendees (comma-separated)",
        "whisper_model": "Whisper Model Size",
        "model_help": "Larger models are more accurate but slower. Base recommended for 500 users.",
        "generate_report": "🚀 Generate Report",
        "processing": "Processing your meeting...",
        "step_transcribing": "Step 1/6: Transcribing audio",
        "step_summarizing": "Step 2/6: Generating summary",
        "step_insights": "Step 3/6: Extracting insights",
        "step_actions": "Step 4/6: Identifying action items",
        "step_takeaways": "Step 5/6: Identifying key takeaways",
        "step_formatting": "Step 6/6: Formatting report",
        "success": "✅ Report Generated Successfully!",
        "download_section": "📥 Download Report",
        "download_txt": "Download as TXT",
        "download_md": "Download as Markdown",
        "download_pdf": "Download as PDF",
        "download_json": "Download as JSON",
        "error": "❌ Error",
        "upload_file_first": "Please upload an audio file first.",
        "preview": "📊 Report Preview",
        "stats": "📈 Statistics",
        "duration": "Duration",
        "segments": "Segments",
        "words": "Words",
        "action_items": "Action Items",
        "takeaways": "Key Takeaways",
    },
    "es": {
        "title": "🎙️ Transcriptor y Generador de Informes de Reuniones",
        "subtitle": "Análisis de Reuniones con IA usando Método de Esquema",
        "language": "Idioma",
        "upload_audio": "Subir Archivo de Audio",
        "audio_help": "Formatos soportados: MP3, WAV, M4A, WebM, MP4, FLAC, OGG (Max 500MB)",
        "meeting_details": "Detalles de la Reunión (Opcional)",
        "meeting_title": "Título de la Reunión",
        "meeting_date": "Fecha de la Reunión",
        "meeting_time": "Hora de la Reunión",
        "location": "Ubicación",
        "organizer": "Organizador",
        "attendees": "Asistentes (separados por comas)",
        "whisper_model": "Tamaño del Modelo Whisper",
        "model_help": "Modelos más grandes son más precisos pero más lentos. Base recomendado para 500 usuarios.",
        "generate_report": "🚀 Generar Informe",
        "processing": "Procesando su reunión...",
        "step_transcribing": "Paso 1/6: Transcribiendo audio",
        "step_summarizing": "Paso 2/6: Generando resumen",
        "step_insights": "Paso 3/6: Extrayendo información",
        "step_actions": "Paso 4/6: Identificando acciones",
        "step_takeaways": "Paso 5/6: Identificando puntos clave",
        "step_formatting": "Paso 6/6: Formateando informe",
        "success": "✅ ¡Informe Generado Exitosamente!",
        "download_section": "📥 Descargar Informe",
        "download_txt": "Descargar como TXT",
        "download_md": "Descargar como Markdown",
        "download_pdf": "Descargar como PDF",
        "download_json": "Descargar como JSON",
        "error": "❌ Error",
        "upload_file_first": "Por favor, suba un archivo de audio primero.",
        "preview": "📊 Vista Previa del Informe",
        "stats": "📈 Estadísticas",
        "duration": "Duración",
        "segments": "Segmentos",
        "words": "Palabras",
        "action_items": "Items de Acción",
        "takeaways": "Puntos Clave",
    },
    "fr": {
        "title": "🎙️ Transcripteur et Générateur de Rapports de Réunion",
        "subtitle": "Analyse de Réunions par IA avec Méthode de Plan",
        "language": "Langue",
        "upload_audio": "Télécharger un Fichier Audio",
        "audio_help": "Formats supportés: MP3, WAV, M4A, WebM, MP4, FLAC, OGG (Max 500MB)",
        "meeting_details": "Détails de la Réunion (Optionnel)",
        "meeting_title": "Titre de la Réunion",
        "meeting_date": "Date de la Réunion",
        "meeting_time": "Heure de la Réunion",
        "location": "Lieu",
        "organizer": "Organisateur",
        "attendees": "Participants (séparés par des virgules)",
        "whisper_model": "Taille du Modèle Whisper",
        "model_help": "Les grands modèles sont plus précis mais plus lents. Base recommandé pour 500 utilisateurs.",
        "generate_report": "🚀 Générer le Rapport",
        "processing": "Traitement de votre réunion...",
        "step_transcribing": "Étape 1/6: Transcription audio",
        "step_summarizing": "Étape 2/6: Génération du résumé",
        "step_insights": "Étape 3/6: Extraction des informations",
        "step_actions": "Étape 4/6: Identification des actions",
        "step_takeaways": "Étape 5/6: Identification des points clés",
        "step_formatting": "Étape 6/6: Formatage du rapport",
        "success": "✅ Rapport Généré avec Succès!",
        "download_section": "📥 Télécharger le Rapport",
        "download_txt": "Télécharger en TXT",
        "download_md": "Télécharger en Markdown",
        "download_pdf": "Télécharger en PDF",
        "download_json": "Télécharger en JSON",
        "error": "❌ Erreur",
        "upload_file_first": "Veuillez d'abord télécharger un fichier audio.",
        "preview": "📊 Aperçu du Rapport",
        "stats": "📈 Statistiques",
        "duration": "Durée",
        "segments": "Segments",
        "words": "Mots",
        "action_items": "Actions à Faire",
        "takeaways": "Points Clés",
    },
    "zh": {
        "title": "🎙️ 会议转录与报告生成器",
        "subtitle": "AI驱动的会议分析（大纲法）",
        "language": "语言",
        "upload_audio": "上传音频文件",
        "audio_help": "支持格式：MP3、WAV、M4A、WebM、MP4、FLAC、OGG（最大500MB）",
        "meeting_details": "会议详情（可选）",
        "meeting_title": "会议标题",
        "meeting_date": "会议日期",
        "meeting_time": "会议时间",
        "location": "地点",
        "organizer": "组织者",
        "attendees": "参与者（逗号分隔）",
        "whisper_model": "Whisper模型大小",
        "model_help": "较大的模型更准确但更慢。推荐使用Base模型支持500用户。",
        "generate_report": "🚀 生成报告",
        "processing": "正在处理您的会议...",
        "step_transcribing": "步骤1/6：转录音频",
        "step_summarizing": "步骤2/6：生成摘要",
        "step_insights": "步骤3/6：提取见解",
        "step_actions": "步骤4/6：识别行动项",
        "step_takeaways": "步骤5/6：识别关键要点",
        "step_formatting": "步骤6/6：格式化报告",
        "success": "✅ 报告生成成功！",
        "download_section": "📥 下载报告",
        "download_txt": "下载TXT格式",
        "download_md": "下载Markdown格式",
        "download_pdf": "下载PDF格式",
        "download_json": "下载JSON格式",
        "error": "❌ 错误",
        "upload_file_first": "请先上传音频文件。",
        "preview": "📊 报告预览",
        "stats": "📈 统计信息",
        "duration": "时长",
        "segments": "片段",
        "words": "字数",
        "action_items": "行动项",
        "takeaways": "关键要点",
    },
    "de": {
        "title": "🎙️ Meeting-Transkriptions- und Berichtsgenerator",
        "subtitle": "KI-gestützte Meeting-Analyse mit Gliederungsmethode",
        "language": "Sprache",
        "upload_audio": "Audiodatei Hochladen",
        "audio_help": "Unterstützte Formate: MP3, WAV, M4A, WebM, MP4, FLAC, OGG (Max 500MB)",
        "meeting_details": "Meeting-Details (Optional)",
        "meeting_title": "Meeting-Titel",
        "meeting_date": "Meeting-Datum",
        "meeting_time": "Meeting-Zeit",
        "location": "Ort",
        "organizer": "Organisator",
        "attendees": "Teilnehmer (durch Kommas getrennt)",
        "whisper_model": "Whisper-Modellgröße",
        "model_help": "Größere Modelle sind genauer, aber langsamer. Base empfohlen für 500 Benutzer.",
        "generate_report": "🚀 Bericht Erstellen",
        "processing": "Ihr Meeting wird verarbeitet...",
        "step_transcribing": "Schritt 1/6: Audio transkribieren",
        "step_summarizing": "Schritt 2/6: Zusammenfassung erstellen",
        "step_insights": "Schritt 3/6: Erkenntnisse extrahieren",
        "step_actions": "Schritt 4/6: Aktionspunkte identifizieren",
        "step_takeaways": "Schritt 5/6: Kernpunkte identifizieren",
        "step_formatting": "Schritt 6/6: Bericht formatieren",
        "success": "✅ Bericht Erfolgreich Erstellt!",
        "download_section": "📥 Bericht Herunterladen",
        "download_txt": "Als TXT Herunterladen",
        "download_md": "Als Markdown Herunterladen",
        "download_pdf": "Als PDF Herunterladen",
        "download_json": "Als JSON Herunterladen",
        "error": "❌ Fehler",
        "upload_file_first": "Bitte laden Sie zuerst eine Audiodatei hoch.",
        "preview": "📊 Berichtsvorschau",
        "stats": "📈 Statistiken",
        "duration": "Dauer",
        "segments": "Segmente",
        "words": "Wörter",
        "action_items": "Aktionspunkte",
        "takeaways": "Kernpunkte",
    },
    "bn": {
        "title": "🎙️ মিটিং ট্রান্সক্রিপশন ও রিপোর্ট জেনারেটর",
        "subtitle": "এআই-চালিত মিটিং বিশ্লেষণ (আউটলাইন পদ্ধতি)",
        "language": "ভাষা",
        "upload_audio": "অডিও ফাইল আপলোড করুন",
        "audio_help": "সমর্থিত ফর্ম্যাট: MP3, WAV, M4A, WebM, MP4, FLAC, OGG (সর্বোচ্চ 500MB)",
        "meeting_details": "মিটিং বিবরণ (ঐচ্ছিক)",
        "meeting_title": "মিটিং শিরোনাম",
        "meeting_date": "মিটিং তারিখ",
        "meeting_time": "মিটিং সময়",
        "location": "স্থান",
        "organizer": "আয়োজক",
        "attendees": "উপস্থিতি (কমা দ্বারা পৃথক)",
        "whisper_model": "Whisper মডেল আকার",
        "model_help": "বড় মডেল আরো নির্ভুল কিন্তু ধীর। ৫০০ ব্যবহারকারীর জন্য Base সুপারিশকৃত।",
        "generate_report": "🚀 রিপোর্ট তৈরি করুন",
        "processing": "আপনার মিটিং প্রক্রিয়াকরণ হচ্ছে...",
        "step_transcribing": "ধাপ ১/৬: অডিও ট্রান্সক্রিপশন",
        "step_summarizing": "ধাপ ২/৬: সারসংক্ষেপ তৈরি",
        "step_insights": "ধাপ ৩/৬: অন্তর্দৃষ্টি উত্তোলন",
        "step_actions": "ধাপ ৪/৬: কর্ম আইটেম চিহ্নিতকরণ",
        "step_takeaways": "ধাপ ৫/৬: মূল বিষয় চিহ্নিতকরণ",
        "step_formatting": "ধাপ ৬/৬: রিপোর্ট ফরম্যাটিং",
        "success": "✅ রিপোর্ট সফলভাবে তৈরি হয়েছে!",
        "download_section": "📥 রিপোর্ট ডাউনলোড করুন",
        "download_txt": "TXT হিসেবে ডাউনলোড",
        "download_md": "Markdown হিসেবে ডাউনলোড",
        "download_pdf": "PDF হিসেবে ডাউনলোড",
        "download_json": "JSON হিসেবে ডাউনলোড",
        "error": "❌ ত্রুটি",
        "upload_file_first": "প্রথমে একটি অডিও ফাইল আপলোড করুন।",
        "preview": "📊 রিপোর্ট প্রিভিউ",
        "stats": "📈 পরিসংখ্যান",
        "duration": "সময়কাল",
        "segments": "অংশসমূহ",
        "words": "শব্দ",
        "action_items": "কর্ম আইটেম",
        "takeaways": "মূল বিষয়",
    }
}

def t(key, lang="en"):
    """Translation helper function"""
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)


# ============================================================
# OPTIMIZED TRANSCRIPTION MODULE (For Concurrent Users)
# ============================================================

@st.cache_resource
def load_whisper_model(model_size="base"):
    """Load and cache Whisper model - shared across users"""
    return whisper.load_model(model_size)

class AudioTranscriber:
    """Handles audio transcription with progress tracking"""
    
    @staticmethod
    def transcribe(audio_path, model_size="base", progress_callback=None):
        """Transcribe with progress updates"""
        model = load_whisper_model(model_size)
        
        if progress_callback:
            progress_callback(0.3, "Loading audio file...")
        
        result = model.transcribe(audio_path, verbose=False)
        
        if progress_callback:
            progress_callback(1.0, "Transcription complete")
        
        return result
    
    @staticmethod
    def format_timestamp(seconds):
        """Convert seconds to HH:MM:SS format"""
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{mins:02d}:{secs:02d}"
        return f"{mins:02d}:{secs:02d}"


# ============================================================
# OPTIMIZED AI ANALYSIS MODULE
# ============================================================

@st.cache_resource
def load_ai_models():
    """Load and cache all AI models - shared across users"""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return summarizer, qa_model, classifier

class MeetingAnalyzer:
    """Analyzes meeting transcripts with caching"""
    
    def __init__(self):
        self.summarizer, self.qa_model, self.classifier = load_ai_models()
    
    def summarize_text(self, text, max_length=150, progress_callback=None):
        """Generate summary with chunking for long texts"""
        chunk_size = 1024
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        summaries = []
        for i, chunk in enumerate(chunks):
            if len(chunk.split()) > 50:
                if progress_callback:
                    progress = (i + 1) / len(chunks)
                    progress_callback(progress, f"Summarizing chunk {i+1}/{len(chunks)}")
                
                summary = self.summarizer(chunk, max_length=max_length, min_length=30, do_sample=False)
                summaries.append(summary[0]["summary_text"])
        
        return " ".join(summaries)
    
    def extract_insights(self, text, progress_callback=None):
        """Extract key insights using Q&A"""
        questions = {
            "objective": "What is the main purpose or goal of this meeting?",
            "decisions": "What decisions were made in this meeting?",
            "concerns": "What problems, issues, or concerns were discussed?",
            "next_steps": "What are the next steps or follow-up actions?",
            "deadlines": "What deadlines or dates were mentioned?",
            "owners": "Who is responsible for tasks or action items?"
        }
        
        insights = {}
        context = text[:4096]
        
        for i, (key, question) in enumerate(questions.items()):
            if progress_callback:
                progress = (i + 1) / len(questions)
                progress_callback(progress, f"Extracting: {key}")
            
            try:
                answer = self.qa_model(question=question, context=context)
                insights[key] = answer["answer"] if answer["score"] > 0.1 else "Not clearly identified"
            except:
                insights[key] = "Not identified"
        
        return insights
    
    def extract_action_items(self, text, progress_callback=None):
        """Extract and prioritize action items"""
        action_keywords = [
            "need to", "should", "will", "must", "have to",
            "action item", "follow up", "deadline", "by next",
            "responsible", "assigned to", "take care of"
        ]
        
        sentences = text.replace("?", ".").replace("!", ".").split(".")
        
        potential_actions = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
                lower = sentence.lower()
                if any(kw in lower for kw in action_keywords):
                    potential_actions.append(sentence)
        
        priority_labels = ["urgent high priority", "medium priority", "low priority"]
        prioritized = []
        
        for i, action in enumerate(potential_actions[:10]):
            if progress_callback:
                progress = (i + 1) / min(len(potential_actions), 10)
                progress_callback(progress, f"Analyzing action {i+1}")
            
            try:
                result = self.classifier(action, priority_labels)
                priority = "HIGH" if "urgent" in result["labels"][0] or "high" in result["labels"][0] else \
                          "MEDIUM" if "medium" in result["labels"][0] else "LOW"
                
                prioritized.append({
                    "task": action,
                    "priority": priority,
                    "confidence": result["scores"][0]
                })
            except:
                prioritized.append({"task": action, "priority": "MEDIUM", "confidence": 0.5})
        
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        prioritized.sort(key=lambda x: priority_order.get(x["priority"], 1))
        
        return prioritized
    
    def identify_key_takeaways(self, text, num_takeaways=5, progress_callback=None):
        """Identify most important points"""
        sentences = text.replace("?", ".").replace("!", ".").split(".")
        sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
        
        importance_labels = ["very important key point", "moderately important", "not important"]
        scored = []
        
        for i, sentence in enumerate(sentences[:30]):
            if progress_callback:
                progress = (i + 1) / min(len(sentences), 30)
                progress_callback(progress, f"Analyzing sentence {i+1}")
            
            try:
                result = self.classifier(sentence, importance_labels)
                if result["labels"][0] == "very important key point":
                    scored.append({"text": sentence, "score": result["scores"][0]})
            except:
                continue
        
        scored.sort(key=lambda x: x["score"], reverse=True)
        return [item["text"] for item in scored[:num_takeaways]]


# ============================================================
# REPORT GENERATOR WITH MULTIPLE FORMAT SUPPORT
# ============================================================

class ReportGenerator:
    """Generates reports in multiple formats"""
    
    PRIORITY_SYMBOLS = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    
    def __init__(self, lang="en"):
        self.lang = lang
    
    def generate_text_report(self, meeting_info, summary, insights, action_items, takeaways, transcription):
        """Generate text format report"""
        separator = "─" * 70
        
        report = f"""
{'='*70}
                         MEETING REPORT
                     (Outline Method Format)
{'='*70}

{separator}
📋 MEETING BASICS
{separator}

    📌 Title:       {meeting_info.get('title', 'N/A')}
    📅 Date:        {meeting_info.get('date', 'N/A')}
    🕐 Time:        {meeting_info.get('time', 'N/A')}
    📍 Location:    {meeting_info.get('location', 'N/A')}
    👤 Organizer:   {meeting_info.get('organizer', 'N/A')}
    👥 Attendees:   {meeting_info.get('attendees', 'N/A')}

{separator}
🎯 MEETING OBJECTIVE
{separator}

    {insights.get('objective', 'Not identified')}

{separator}
📝 EXECUTIVE SUMMARY
{separator}

    {self._wrap_text(summary, 66)}

{separator}
⭐ KEY TAKEAWAYS
{separator}

"""
        if takeaways:
            for i, takeaway in enumerate(takeaways, 1):
                report += f"    {i}. {self._wrap_text(takeaway, 62)}\n\n"
        else:
            report += "    No key takeaways identified.\n"
        
        report += f"""
{separator}
✅ DECISIONS MADE
{separator}

    {insights.get('decisions', 'No decisions identified')}

{separator}
⚠️ CONCERNS & ISSUES RAISED
{separator}

    {insights.get('concerns', 'No concerns identified')}

{separator}
📋 ACTION ITEMS (Prioritized)
{separator}

    Legend: 🔴 High Priority  🟡 Medium Priority  🟢 Low Priority

"""
        if action_items:
            for i, item in enumerate(action_items, 1):
                symbol = self.PRIORITY_SYMBOLS.get(item["priority"], "⚪")
                priority = item["priority"]
                task = item["task"]
                report += f"    {symbol} [{priority:6}] {i}. {task}\n\n"
        else:
            report += "    No specific action items identified.\n"
        
        report += f"""
{separator}
📅 NEXT STEPS & FOLLOW-UPS
{separator}

    {insights.get('next_steps', 'Not identified')}

    📅 Deadlines: {insights.get('deadlines', 'None identified')}
    👤 Owners:    {insights.get('owners', 'Not identified')}

{separator}
📜 FULL TRANSCRIPT (With Timestamps)
{separator}

"""
        for segment in transcription["segments"]:
            start = AudioTranscriber.format_timestamp(segment["start"])
            end = AudioTranscriber.format_timestamp(segment["end"])
            text = segment["text"].strip()
            report += f"    [{start} → {end}]  {text}\n"
        
        report += f"""
{'='*70}
                       END OF MEETING REPORT
               Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*70}
"""
        return report
    
    def generate_json_report(self, meeting_info, summary, insights, action_items, takeaways, transcription):
        """Generate JSON format report"""
        return json.dumps({
            "meeting_info": meeting_info,
            "summary": summary,
            "insights": insights,
            "action_items": action_items,
            "takeaways": takeaways,
            "transcript": {
                "text": transcription["text"],
                "segments": transcription["segments"]
            },
            "generated_at": datetime.now().isoformat()
        }, indent=2, ensure_ascii=False)
    
    def generate_pdf_report(self, meeting_info, summary, insights, action_items, takeaways):
        """Generate PDF format report with proper error handling"""
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_left_margin(10)
            pdf.set_right_margin(10)
            
            # Title
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "MEETING REPORT", ln=True, align="C")
            pdf.ln(5)
            
            # Meeting Info
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Meeting Details", ln=True)
            pdf.set_font("Arial", "", 10)
            
            # Safely encode text for PDF (Latin-1 compatible only)
            def safe_encode(text):
                """Convert text to Latin-1 safe format"""
                if not text:
                    return "N/A"
                try:
                    # Try to encode as latin-1, replace unsupported chars
                    return text.encode('latin-1', errors='replace').decode('latin-1')
                except:
                    return str(text).encode('ascii', errors='ignore').decode('ascii')
            
            meeting_details = (
                f"Title: {safe_encode(meeting_info.get('title', 'N/A'))}\n"
                f"Date: {safe_encode(meeting_info.get('date', 'N/A'))}\n"
                f"Time: {safe_encode(meeting_info.get('time', 'N/A'))}\n"
                f"Location: {safe_encode(meeting_info.get('location', 'N/A'))}\n"
                f"Organizer: {safe_encode(meeting_info.get('organizer', 'N/A'))}\n"
                f"Attendees: {safe_encode(meeting_info.get('attendees', 'N/A'))}"
            )
            pdf.multi_cell(0, 5, meeting_details)
            pdf.ln(5)
            
            # Summary
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Executive Summary", ln=True)
            pdf.set_font("Arial", "", 10)
            safe_summary = safe_encode(summary)
            if len(safe_summary) > 500:
                safe_summary = safe_summary[:497] + "..."
            pdf.multi_cell(0, 5, safe_summary)
            pdf.ln(5)
            
            # Key Takeaways
            if takeaways:
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "Key Takeaways", ln=True)
                pdf.set_font("Arial", "", 10)
                for i, takeaway in enumerate(takeaways, 1):
                    safe_takeaway = safe_encode(takeaway)
                    # Limit length to prevent overflow
                    if len(safe_takeaway) > 200:
                        safe_takeaway = safe_takeaway[:197] + "..."
                    pdf.multi_cell(0, 5, f"{i}. {safe_takeaway}")
                    pdf.ln(2)
                pdf.ln(3)
            
            # Action Items
            if action_items:
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "Action Items", ln=True)
                pdf.set_font("Arial", "", 10)
                for i, item in enumerate(action_items, 1):
                    priority = item["priority"]
                    task = safe_encode(item["task"])
                    # Limit length to prevent overflow
                    if len(task) > 200:
                        task = task[:197] + "..."
                    pdf.multi_cell(0, 5, f"[{priority}] {i}. {task}")
                    pdf.ln(2)
                pdf.ln(3)
            
            # Insights
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Key Insights", ln=True)
            pdf.set_font("Arial", "", 10)
            
            for key in ["objective", "decisions", "concerns", "next_steps"]:
                value = safe_encode(insights.get(key, "Not identified"))
                if len(value) > 300:
                    value = value[:297] + "..."
                pdf.multi_cell(0, 5, f"{key.replace('_', ' ').title()}: {value}")
                pdf.ln(2)
            
            return pdf.output(dest='S').encode('latin-1', errors='replace')
            
        except Exception as e:
            # If PDF generation fails, create a simple error PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "PDF Generation Error", ln=True, align="C")
            pdf.ln(10)
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 10, f"Unable to generate PDF report.\nError: {str(e)}\n\nPlease use TXT or Markdown format instead.")
            return pdf.output(dest='S').encode('latin-1', errors='replace')
    
    def _wrap_text(self, text, width):
        """Wrap text to specified width"""
        if not text:
            return "N/A"
        
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return "\n    ".join(lines)


# ============================================================
# STREAMLIT APP
# ============================================================

def main():
    # Page config
    st.set_page_config(
        page_title="Meeting Transcription & Report Generator",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if "report_generated" not in st.session_state:
        st.session_state.report_generated = False
    if "report_data" not in st.session_state:
        st.session_state.report_data = None
    
    # Sidebar - Language Selection
    st.sidebar.title("⚙️ Settings")
    lang = st.sidebar.selectbox(
        "🌍 " + t("language", "en"),
        options=["en", "es", "fr", "zh", "de", "bn"],
        format_func=lambda x: {"en": "English", "es": "Español", "fr": "Français", "zh": "中文", "de": "Deutsch", "bn": "বাংলা"}[x]
    )
    
    # Main title
    st.title(t("title", lang))
    st.markdown(f"**{t('subtitle', lang)}**")
    st.markdown("---")
    
    # File upload
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(t("upload_audio", lang))
        audio_file = st.file_uploader(
            t("audio_help", lang),
            type=["mp3", "wav", "m4a", "webm", "mp4", "flac", "ogg"],
            help=t("audio_help", lang)
        )
        
        # FFmpeg warning
        if audio_file and not audio_file.name.endswith('.wav'):
            st.warning("⚠️ Non-WAV files require FFmpeg. Install FFmpeg or use WAV format for best results.")
    
    with col2:
        st.subheader(t("whisper_model", lang))
        model_size = st.selectbox(
            t("model_help", lang),
            options=["tiny", "base", "small", "medium"],
            index=1,
            help=t("model_help", lang)
        )
    
    # Meeting details
    st.markdown("---")
    st.subheader(t("meeting_details", lang))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        meeting_title = st.text_input(t("meeting_title", lang), value="Weekly Team Sync")
        meeting_date = st.date_input(t("meeting_date", lang), value=datetime.now())
        meeting_time = st.time_input(t("meeting_time", lang), value=datetime.now().time())
    
    with col2:
        location = st.text_input(t("location", lang), value="Zoom Meeting")
        organizer = st.text_input(t("organizer", lang), value="")
    
    with col3:
        attendees = st.text_area(
            t("attendees", lang),
            value="",
            height=100,
            help="Separate multiple attendees with commas"
        )
    
    # Generate button
    st.markdown("---")
    if st.button(t("generate_report", lang), type="primary", use_container_width=True):
        if audio_file is None:
            st.error(t("upload_file_first", lang))
        else:
            # Prepare meeting info
            meeting_info = {
                "title": meeting_title,
                "date": meeting_date.strftime("%Y-%m-%d"),
                "time": meeting_time.strftime("%H:%M"),
                "location": location,
                "organizer": organizer if organizer else "Not specified",
                "attendees": attendees if attendees else "Not specified"
            }
            
            # Process the audio
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tmp_file:
                    tmp_file.write(audio_file.read())
                    tmp_path = tmp_file.name
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Transcription
                status_text.text(t("step_transcribing", lang))
                progress_bar.progress(0.15)
                
                transcription = AudioTranscriber.transcribe(
                    tmp_path,
                    model_size=model_size,
                    progress_callback=lambda p, m: progress_bar.progress(0.15 + p * 0.15)
                )
                transcript_text = transcription["text"]
                
                # Initialize analyzer
                analyzer = MeetingAnalyzer()
                
                # Step 2: Summary
                status_text.text(t("step_summarizing", lang))
                progress_bar.progress(0.35)
                
                summary = analyzer.summarize_text(
                    transcript_text,
                    progress_callback=lambda p, m: progress_bar.progress(0.35 + p * 0.15)
                )
                
                # Step 3: Insights
                status_text.text(t("step_insights", lang))
                progress_bar.progress(0.50)
                
                insights = analyzer.extract_insights(
                    transcript_text,
                    progress_callback=lambda p, m: progress_bar.progress(0.50 + p * 0.15)
                )
                
                # Step 4: Action Items
                status_text.text(t("step_actions", lang))
                progress_bar.progress(0.65)
                
                action_items = analyzer.extract_action_items(
                    transcript_text,
                    progress_callback=lambda p, m: progress_bar.progress(0.65 + p * 0.15)
                )
                
                # Step 5: Key Takeaways
                status_text.text(t("step_takeaways", lang))
                progress_bar.progress(0.80)
                
                takeaways = analyzer.identify_key_takeaways(
                    transcript_text,
                    progress_callback=lambda p, m: progress_bar.progress(0.80 + p * 0.15)
                )
                
                # Step 6: Generate Reports
                status_text.text(t("step_formatting", lang))
                progress_bar.progress(0.95)
                
                generator = ReportGenerator(lang)
                
                # Generate all formats
                text_report = generator.generate_text_report(
                    meeting_info, summary, insights, action_items, takeaways, transcription
                )
                
                json_report = generator.generate_json_report(
                    meeting_info, summary, insights, action_items, takeaways, transcription
                )
                
                # Try to generate PDF, but don't fail if it doesn't work
                try:
                    pdf_report = generator.generate_pdf_report(
                        meeting_info, summary, insights, action_items, takeaways
                    )
                    pdf_available = True
                except Exception as pdf_error:
                    st.warning(f"⚠️ PDF generation failed: {str(pdf_error)}. PDF download will be unavailable.")
                    pdf_report = None
                    pdf_available = False
                
                # Store in session state
                st.session_state.report_data = {
                    "text": text_report,
                    "json": json_report,
                    "pdf": pdf_report,
                    "pdf_available": pdf_available,
                    "meeting_info": meeting_info,
                    "summary": summary,
                    "insights": insights,
                    "action_items": action_items,
                    "takeaways": takeaways,
                    "transcription": transcription,
                    "stats": {
                        "duration": AudioTranscriber.format_timestamp(transcription["segments"][-1]["end"]),
                        "segments": len(transcription["segments"]),
                        "words": len(transcript_text.split()),
                        "action_items": len(action_items),
                        "takeaways": len(takeaways)
                    }
                }
                st.session_state.report_generated = True
                
                # Complete
                progress_bar.progress(1.0)
                status_text.text("")
                progress_bar.empty()
                
                # Clean up temp file
                os.unlink(tmp_path)
                
                st.success(t("success", lang))
                st.rerun()
                
            except Exception as e:
                st.error(f"{t('error', lang)}: {str(e)}")
                st.code(traceback.format_exc())
                # Clean up temp file if exists
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    
    # Display report if generated
    if st.session_state.report_generated and st.session_state.report_data:
        st.markdown("---")
        
        # Statistics
        st.subheader(t("stats", lang))
        stats = st.session_state.report_data["stats"]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric(t("duration", lang), stats["duration"])
        col2.metric(t("segments", lang), stats["segments"])
        col3.metric(t("words", lang), stats["words"])
        col4.metric(t("action_items", lang), stats["action_items"])
        col5.metric(t("takeaways", lang), stats["takeaways"])
        
        # Download section
        st.markdown("---")
        st.subheader(t("download_section", lang))
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.download_button(
                label=t("download_txt", lang),
                data=st.session_state.report_data["text"],
                file_name=f"meeting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                label=t("download_md", lang),
                data=st.session_state.report_data["text"],
                file_name=f"meeting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        with col3:
            if st.session_state.report_data.get("pdf_available", False):
                st.download_button(
                    label=t("download_pdf", lang),
                    data=st.session_state.report_data["pdf"],
                    file_name=f"meeting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.button(
                    label="⚠️ PDF Unavailable",
                    disabled=True,
                    use_container_width=True,
                    help="PDF generation failed. Please use TXT or Markdown format."
                )
        
        with col4:
            st.download_button(
                label=t("download_json", lang),
                data=st.session_state.report_data["json"],
                file_name=f"meeting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Report preview
        st.markdown("---")
        st.subheader(t("preview", lang))
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "📋 Action Items", "⭐ Takeaways", "📜 Transcript"])
        
        with tab1:
            st.markdown("### 🎯 Meeting Objective")
            st.info(st.session_state.report_data["insights"].get("objective", "Not identified"))
            
            st.markdown("### 📝 Executive Summary")
            st.write(st.session_state.report_data["summary"])
            
            st.markdown("### ✅ Decisions Made")
            st.write(st.session_state.report_data["insights"].get("decisions", "No decisions identified"))
            
            st.markdown("### ⚠️ Concerns & Issues")
            st.write(st.session_state.report_data["insights"].get("concerns", "No concerns identified"))
        
        with tab2:
            st.markdown("### 📋 Prioritized Action Items")
            
            if st.session_state.report_data["action_items"]:
                for i, item in enumerate(st.session_state.report_data["action_items"], 1):
                    priority = item["priority"]
                    symbol = ReportGenerator.PRIORITY_SYMBOLS.get(priority, "⚪")
                    
                    if priority == "HIGH":
                        st.error(f"{symbol} **[{priority}]** {i}. {item['task']}")
                    elif priority == "MEDIUM":
                        st.warning(f"{symbol} **[{priority}]** {i}. {item['task']}")
                    else:
                        st.success(f"{symbol} **[{priority}]** {i}. {item['task']}")
            else:
                st.info("No specific action items identified.")
            
            st.markdown("### 📅 Next Steps & Follow-ups")
            st.write(st.session_state.report_data["insights"].get("next_steps", "Not identified"))
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📅 Deadlines:**")
                st.write(st.session_state.report_data["insights"].get("deadlines", "None identified"))
            with col2:
                st.markdown("**👤 Owners/Assignees:**")
                st.write(st.session_state.report_data["insights"].get("owners", "Not identified"))
        
        with tab3:
            st.markdown("### ⭐ Key Takeaways")
            
            if st.session_state.report_data["takeaways"]:
                for i, takeaway in enumerate(st.session_state.report_data["takeaways"], 1):
                    st.markdown(f"**{i}.** {takeaway}")
            else:
                st.info("No key takeaways identified.")
        
        with tab4:
            st.markdown("### 📜 Full Transcript with Timestamps")
            
            # Display transcript in expandable sections
            transcript_text = ""
            for segment in st.session_state.report_data["transcription"]["segments"]:
                start = AudioTranscriber.format_timestamp(segment["start"])
                end = AudioTranscriber.format_timestamp(segment["end"])
                text = segment["text"].strip()
                transcript_text += f"**[{start} → {end}]** {text}\n\n"
            
            st.markdown(transcript_text)
        
        # Full report in expander
        with st.expander("📄 View Full Text Report"):
            st.code(st.session_state.report_data["text"], language=None)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>🎙️ <b>Meeting Transcription & Report Generator</b></p>
        <p>Powered by OpenAI Whisper & HuggingFace Transformers</p>
        <p>Optimized for 500 concurrent users with model caching and efficient resource management</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()