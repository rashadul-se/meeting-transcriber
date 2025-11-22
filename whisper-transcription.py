"""
বাংলা মিটিং ট্রান্সক্রিপশন এবং রিপোর্ট জেনারেটর
==============================================
Bangla Meeting Transcription & Outline Report Generator

Whisper ব্যবহার করে বাংলা অডিও ট্রান্সক্রাইব করে এবং 
Outline Method অনুযায়ী রিপোর্ট তৈরি করে।

Installation:
    pip install openai-whisper transformers torch sentencepiece

Usage:
    1. AUDIO_FILE এ আপনার অডিও ফাইলের পাথ দিন
    2. MEETING_INFO তে মিটিংয়ের তথ্য আপডেট করুন
    3. রান করুন: python bangla_meeting_report.py
"""

import whisper
import os
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# বাংলা টেক্সট কনফিগারেশন
# ============================================================

BANGLA_LABELS = {
    "high_priority": "উচ্চ অগ্রাধিকার",
    "medium_priority": "মাঝারি অগ্রাধিকার", 
    "low_priority": "নিম্ন অগ্রাধিকার",
    "not_identified": "চিহ্নিত করা যায়নি",
    "none": "কোনটি নেই"
}

BANGLA_ACTION_KEYWORDS = [
    "করতে হবে", "করবেন", "করবে", "দরকার", "প্রয়োজন",
    "দায়িত্ব", "জরুরি", "আগামী", "পরবর্তী", "ডেডলাইন",
    "সময়সীমা", "জমা দিতে", "পাঠাতে হবে", "শেষ করতে",
    "need to", "should", "will", "must", "deadline"
]


# ============================================================
# ট্রান্সক্রিপশন মডিউল
# ============================================================

class BanglaAudioTranscriber:
    """বাংলা অডিও ট্রান্সক্রিপশন Whisper দিয়ে।"""
    
    def __init__(self, model_size="medium"):
        """
        ট্রান্সক্রাইবার ইনিশিয়ালাইজ।
        
        Args:
            model_size: "tiny", "base", "small", "medium", "large"
                        বাংলার জন্য "medium" বা "large" ভালো কাজ করে
        """
        self.model_size = model_size
        self.model = None
    
    def load_model(self):
        """Whisper মডেল লোড।"""
        print(f"📥 Whisper {self.model_size} মডেল লোড হচ্ছে...")
        self.model = whisper.load_model(self.model_size)
        print("✅ মডেল সফলভাবে লোড হয়েছে")
    
    def transcribe(self, audio_path, language="bn"):
        """
        অডিও ফাইল ট্রান্সক্রাইব।
        
        Args:
            audio_path: অডিও ফাইলের পাথ
            language: "bn" বাংলার জন্য, "en" ইংরেজির জন্য
        
        Returns:
            dict with 'text' and 'segments'
        """
        if self.model is None:
            self.load_model()
        
        print(f"🎙️ ট্রান্সক্রাইব হচ্ছে: {audio_path}")
        
        result = self.model.transcribe(
            audio_path,
            language=language,
            task="transcribe"
        )
        
        print(f"✅ ট্রান্সক্রিপশন সম্পন্ন ({len(result['segments'])} সেগমেন্ট)")
        return result
    
    @staticmethod
    def format_timestamp(seconds):
        """সেকেন্ডকে MM:SS ফরম্যাটে রূপান্তর।"""
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{mins:02d}:{secs:02d}"
        return f"{mins:02d}:{secs:02d}"


# ============================================================
# বাংলা টেক্সট এনালাইসিস মডিউল
# ============================================================

class BanglaMeetingAnalyzer:
    """বাংলা মিটিং ট্রান্সক্রিপ্ট বিশ্লেষণ।"""
    
    def __init__(self):
        self.summarizer = None
        self.classifier = None
    
    def load_models(self):
        """AI মডেল লোড।"""
        print("📥 AI মডেল লোড হচ্ছে...")
        
        # mT5 multilingual summarizer (supports Bangla)
        print("   - সারাংশ মডেল লোড হচ্ছে...")
        self.summarizer = pipeline(
            "summarization",
            model="csebuetnlp/mT5_multilingual_XLSum"
        )
        
        # Multilingual classifier
        print("   - শ্রেণীবিভাগ মডেল লোড হচ্ছে...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli"
        )
        
        print("✅ সব মডেল লোড হয়েছে")
    
    def summarize_text(self, text, max_length=200):
        """বাংলা টেক্সট সারাংশ।"""
        if self.summarizer is None:
            self.load_models()
        
        # Chunk text for processing
        chunk_size = 512
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        summaries = []
        for i, chunk in enumerate(chunks):
            if len(chunk.split()) > 30:
                print(f"   সারাংশ তৈরি হচ্ছে {i+1}/{len(chunks)}...")
                try:
                    summary = self.summarizer(
                        chunk,
                        max_length=max_length,
                        min_length=30,
                        do_sample=False
                    )
                    summaries.append(summary[0]["summary_text"])
                except:
                    continue
        
        return " ".join(summaries) if summaries else text[:500]
    
    def extract_action_items(self, text):
        """অ্যাকশন আইটেম বের করা এবং প্রায়োরিটি দেওয়া।"""
        if self.classifier is None:
            self.load_models()
        
        # Split into sentences
        sentences = text.replace("।", ".").replace("?", ".").replace("!", ".").split(".")
        
        # Filter potential action items
        potential_actions = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15:
                lower = sentence.lower()
                if any(kw in lower or kw in sentence for kw in BANGLA_ACTION_KEYWORDS):
                    potential_actions.append(sentence)
        
        # Classify priority
        priority_labels = [
            "very urgent important",
            "moderately important", 
            "not urgent low priority"
        ]
        
        prioritized = []
        for action in potential_actions[:10]:
            try:
                result = self.classifier(action, priority_labels)
                label = result["labels"][0]
                
                if "urgent" in label:
                    priority = "HIGH"
                    priority_bn = BANGLA_LABELS["high_priority"]
                elif "moderate" in label:
                    priority = "MEDIUM"
                    priority_bn = BANGLA_LABELS["medium_priority"]
                else:
                    priority = "LOW"
                    priority_bn = BANGLA_LABELS["low_priority"]
                
                prioritized.append({
                    "task": action,
                    "priority": priority,
                    "priority_bn": priority_bn,
                    "confidence": result["scores"][0]
                })
            except:
                prioritized.append({
                    "task": action,
                    "priority": "MEDIUM",
                    "priority_bn": BANGLA_LABELS["medium_priority"],
                    "confidence": 0.5
                })
        
        # Sort by priority
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        prioritized.sort(key=lambda x: priority_order.get(x["priority"], 1))
        
        return prioritized
    
    def identify_key_points(self, text, num_points=5):
        """গুরুত্বপূর্ণ পয়েন্ট চিহ্নিত করা।"""
        if self.classifier is None:
            self.load_models()
        
        sentences = text.replace("।", ".").split(".")
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        importance_labels = ["very important", "somewhat important", "not important"]
        
        scored = []
        for sentence in sentences[:25]:
            try:
                result = self.classifier(sentence, importance_labels)
                if result["labels"][0] == "very important":
                    scored.append({
                        "text": sentence,
                        "score": result["scores"][0]
                    })
            except:
                continue
        
        scored.sort(key=lambda x: x["score"], reverse=True)
        return [item["text"] for item in scored[:num_points]]


# ============================================================
# বাংলা রিপোর্ট জেনারেটর
# ============================================================

class BanglaOutlineReportGenerator:
    """Outline Method অনুযায়ী বাংলা মিটিং রিপোর্ট।"""
    
    PRIORITY_SYMBOLS = {
        "HIGH": "🔴",
        "MEDIUM": "🟡",
        "LOW": "🟢"
    }
    
    def __init__(self, model_size="medium"):
        """
        Args:
            model_size: Whisper মডেল সাইজ
                        বাংলার জন্য "medium" বা "large" রেকমেন্ডেড
        """
        self.transcriber = BanglaAudioTranscriber(model_size)
        self.analyzer = BanglaMeetingAnalyzer()
    
    def generate_report(self, audio_path, meeting_info=None, language="bn"):
        """
        সম্পূর্ণ মিটিং রিপোর্ট তৈরি।
        
        Args:
            audio_path: অডিও ফাইলের পাথ
            meeting_info: মিটিংয়ের তথ্য
            language: "bn" বাংলা, "en" ইংরেজি
        """
        if meeting_info is None:
            meeting_info = self._default_meeting_info()
        
        print("\n" + "="*60)
        print("        মিটিং রিপোর্ট তৈরি হচ্ছে")
        print("="*60)
        
        # Step 1: Transcribe
        print("\n📌 ধাপ ১: অডিও ট্রান্সক্রাইব হচ্ছে...")
        transcription = self.transcriber.transcribe(audio_path, language)
        transcript_text = transcription["text"]
        
        # Step 2: Summarize
        print("\n📌 ধাপ ২: সারাংশ তৈরি হচ্ছে...")
        summary = self.analyzer.summarize_text(transcript_text)
        
        # Step 3: Extract action items
        print("\n📌 ধাপ ৩: কর্ম পরিকল্পনা চিহ্নিত হচ্ছে...")
        action_items = self.analyzer.extract_action_items(transcript_text)
        
        # Step 4: Key points
        print("\n📌 ধাপ ৪: মূল পয়েন্ট চিহ্নিত হচ্ছে...")
        key_points = self.analyzer.identify_key_points(transcript_text)
        
        # Step 5: Generate report
        print("\n📌 ধাপ ৫: রিপোর্ট তৈরি হচ্ছে...")
        report = self._format_report(
            meeting_info=meeting_info,
            summary=summary,
            action_items=action_items,
            key_points=key_points,
            transcription=transcription
        )
        
        print("\n✅ রিপোর্ট তৈরি সম্পন্ন!")
        return report
    
    def _default_meeting_info(self):
        """ডিফল্ট মিটিং তথ্য।"""
        return {
            "title": "মিটিং",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M"),
            "location": "উল্লেখ নেই",
            "organizer": "উল্লেখ নেই",
            "attendees": ["উল্লেখ নেই"]
        }
    
    def _format_report(self, meeting_info, summary, action_items, 
                       key_points, transcription):
        """রিপোর্ট ফরম্যাট।"""
        
        sep = "━" * 70
        
        report = f"""
{'='*70}
                         মিটিং রিপোর্ট
                    (আউটলাইন মেথড ফরম্যাট)
                      MEETING REPORT
{'='*70}

{sep}
📋 মিটিংয়ের মূল তথ্য (পাঁচটি W)
{sep}

    📌 শিরোনাম:      {meeting_info.get('title', 'N/A')}
    📅 তারিখ:        {meeting_info.get('date', 'N/A')}
    🕐 সময়:         {meeting_info.get('time', 'N/A')}
    📍 স্থান:        {meeting_info.get('location', 'N/A')}
    👤 আয়োজক:       {meeting_info.get('organizer', 'N/A')}
    👥 অংশগ্রহণকারী:  {', '.join(meeting_info.get('attendees', ['N/A']))}

{sep}
📝 সারাংশ (Executive Summary)
{sep}

    {summary}

{sep}
⭐ মূল পয়েন্ট / Key Takeaways
{sep}

"""
        # Key points
        if key_points:
            for i, point in enumerate(key_points, 1):
                report += f"    {i}। {point}\n\n"
        else:
            report += f"    {BANGLA_LABELS['not_identified']}\n"
        
        report += f"""
{sep}
📋 কর্ম পরিকল্পনা (অগ্রাধিকার অনুযায়ী)
    Action Items (Prioritized)
{sep}

    চিহ্ন: 🔴 উচ্চ অগ্রাধিকার  🟡 মাঝারি  🟢 নিম্ন
    Legend: 🔴 High  🟡 Medium  🟢 Low

"""
        # Action items
        if action_items:
            for i, item in enumerate(action_items, 1):
                symbol = self.PRIORITY_SYMBOLS.get(item["priority"], "⚪")
                priority_bn = item["priority_bn"]
                task = item["task"]
                report += f"    {symbol} [{priority_bn}]\n"
                report += f"       {i}। {task}\n\n"
        else:
            report += f"    {BANGLA_LABELS['not_identified']}\n"
        
        report += f"""
{sep}
🔜 পরবর্তী পদক্ষেপ / Next Steps
{sep}

    মিটিংয়ে আলোচিত পরবর্তী পদক্ষেপগুলো উপরের কর্ম পরিকল্পনায় 
    অগ্রাধিকার অনুযায়ী সাজানো হয়েছে।

{sep}
📜 সম্পূর্ণ ট্রান্সক্রিপ্ট (Full Transcript)
{sep}

"""
        # Transcript
        for seg in transcription["segments"]:
            start = BanglaAudioTranscriber.format_timestamp(seg["start"])
            end = BanglaAudioTranscriber.format_timestamp(seg["end"])
            text = seg["text"].strip()
            report += f"    [{start} → {end}]  {text}\n"
        
        report += f"""
{'='*70}
                      রিপোর্ট সমাপ্ত
                    END OF REPORT
           তৈরির সময়: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*70}
"""
        return report
    
    def save_report(self, report, output_path):
        """রিপোর্ট ফাইলে সেভ।"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"💾 রিপোর্ট সেভ হয়েছে: {output_path}")


# ============================================================
# মূল প্রোগ্রাম
# ============================================================

if __name__ == "__main__":
    
    # ========== কনফিগারেশন ==========
    
    # অডিও ফাইলের পাথ (আপডেট করুন)
    AUDIO_FILE = "meeting.mp3"
    
    # আউটপুট ফাইলের পাথ
    OUTPUT_FILE = "meeting_report_bangla.txt"
    
    # ভাষা: "bn" বাংলা, "en" ইংরেজি, "mixed" মিশ্র
    LANGUAGE = "bn"
    
    # Whisper মডেল সাইজ
    # বাংলার জন্য "medium" বা "large" ভালো কাজ করে
    MODEL_SIZE = "medium"
    
    # মিটিংয়ের তথ্য (আপডেট করুন)
    MEETING_INFO = {
        "title": "সাপ্তাহিক টিম মিটিং",
        "date": "২০২৪-০১-১৫",
        "time": "সকাল ১০:০০",
        "location": "জুম মিটিং",
        "organizer": "রহিম সাহেব",
        "attendees": [
            "রহিম সাহেব",
            "করিম ভাই",
            "ফাতেমা আপা",
            "সালমা বেগম",
            "জাহিদ হাসান"
        ]
    }
    
    # ========== রিপোর্ট তৈরি ==========
    
    if not os.path.exists(AUDIO_FILE):
        print(f"""
❌ ত্রুটি: অডিও ফাইল '{AUDIO_FILE}' পাওয়া যায়নি।

অনুগ্রহ করে AUDIO_FILE ভেরিয়েবলে সঠিক পাথ দিন।

সাপোর্টেড ফরম্যাট: mp3, wav, m4a, webm, mp4, flac, ogg

উদাহরণ:
    AUDIO_FILE = "/path/to/your/meeting.mp3"
    AUDIO_FILE = "recording.wav"
""")
    else:
        # জেনারেটর তৈরি
        generator = BanglaOutlineReportGenerator(model_size=MODEL_SIZE)
        
        # রিপোর্ট তৈরি
        report = generator.generate_report(
            AUDIO_FILE, 
            MEETING_INFO,
            language=LANGUAGE
        )
        
        # কনসোলে প্রিন্ট
        print(report)
        
        # ফাইলে সেভ
        generator.save_report(report, OUTPUT_FILE)
        
        print("\n" + "="*60)
        print("✅ সম্পন্ন! আপনার মিটিং রিপোর্ট তৈরি হয়েছে।")
        print("="*60)
