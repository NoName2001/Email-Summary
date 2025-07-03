import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font
import os
import base64
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from datetime import datetime
import re
import html
import unicodedata
from bs4 import BeautifulSoup
import threading
import html2text
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from heapq import nlargest
from underthesea import word_tokenize as vi_tokenize
import concurrent.futures


class Constants:
    FONT_SIZE_NORMAL = 10
    FONT_SIZE_BUTTON = 11
    FONT_SIZE_TITLE = 20
    DEFAULT_EMAIL_COUNT = 10
    MAX_EMAIL_COUNT = 50
    MIN_EMAIL_COUNT = 1
    DEFAULT_SUMMARY_SENTENCES = 3
    MAX_SUMMARY_SENTENCES = 10
    MIN_SUMMARY_SENTENCES = 1
    BUTTON_WIDTH = 15
    SPINBOX_WIDTH = 5
    PROGRESS_LENGTH = 300
    WINDOW_SIZE = "1300x800"
    COLUMN_NO_WIDTH = 50
    COLUMN_FROM_WIDTH = 200
    COLUMN_SUBJECT_WIDTH = 300
    COLUMN_DATE_WIDTH = 150
    PADDING = 10
    SMALL_PADDING = 5
    LARGE_PADDING = 20
    ROW_HEIGHT = 30
    WRAP_LENGTH = 600
    FONT_FAMILY = 'Segoe UI'
    COLORS = {
        'primary': '#1976D2',
        'primary_light': '#1E88E5',
        'disabled': '#90CAF9',
        'text_button': '#000000',
        'text_normal': '#212121',
        'text_white': '#FFFFFF',
        'background': '#F5F5F5',
        'link': '#0000FF'
    }


class ResourceLoader:
    @staticmethod
    def init_nltk():
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')


class VietnameseTextSummarizer:
    def __init__(self):
        self.vietnamese_stop_words = {
            'bị', 'bởi', 'cả', 'các', 'cái', 'cần', 'càng', 'chỉ', 'chiếc',
            'cho', 'chứ', 'chưa', 'chuyện', 'có', 'có thể', 'cứ', 'của', 'cùng',
            'cũng', 'đã', 'đang', 'để', 'đến nỗi', 'đều', 'điều', 'do', 'đó',
            'được', 'dưới', 'gì', 'khi', 'không', 'là', 'lại', 'lên', 'lúc',
            'mà', 'mỗi', 'một cách', 'này', 'nên', 'nếu', 'ngay', 'nhiều',
            'như', 'nhưng', 'những', 'nơi', 'nữa', 'phải', 'qua', 'ra', 'rằng',
            'rất', 'rồi', 'sau', 'sẽ', 'so', 'sự', 'tại', 'theo', 'thì', 'trên',
            'trước', 'từ', 'từng', 'và', 'vẫn', 'vào', 'vậy', 'vì', 'việc',
            'với', 'vừa'
        }
        self.stop_words = self.vietnamese_stop_words.union(set(punctuation))

    def preprocess_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = text.strip()
        return text

    def split_sentences(self, text):
        text = self.preprocess_text(text)
        sentence_endings = r'([.!?]|\.{3})+(?=\s|$)'
        sentences = re.split(sentence_endings, text)
        sentences = [s.strip() for s in sentences if s and s.strip()]
        return sentences

    def split_words(self, sentence):
        words = vi_tokenize(sentence.lower())
        return [w for w in words if w not in self.stop_words]

    def summarize(self, text, num_sentences=3):
        try:
            if not text or not text.strip():
                return "Không có nội dung để tóm tắt."

            sentences = self.split_sentences(text)
            if len(sentences) <= num_sentences:
                return text

            word_freq = {}
            for sentence in sentences:
                for word in self.split_words(sentence):
                    word_freq[word] = word_freq.get(word, 0) + 1

            sentence_scores = {}
            for sentence in sentences:
                if len(sentence.split()) < 4:
                    continue
                for word in self.split_words(sentence):
                    if word in word_freq:
                        if sentence not in sentence_scores:
                            sentence_scores[sentence] = word_freq[word]
                        else:
                            sentence_scores[sentence] += word_freq[word]

            summary_sentences = nlargest(
                num_sentences, sentence_scores, key=sentence_scores.get)
            summary_sentences.sort(key=lambda x: sentences.index(x))

            return ' '.join(summary_sentences)
        except Exception as e:
            print(f"Lỗi khi tóm tắt: {str(e)}")
            return text


class GmailAnalyzer:
    def __init__(self):
        self.SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
        self.service = None
        self.summarizer = VietnameseTextSummarizer()

    def connect(self):
        try:
            creds = None
            if os.path.exists('token.json'):
                creds = Credentials.from_authorized_user_file(
                    'token.json', self.SCOPES)
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        'credentials.json', self.SCOPES)
                    creds = flow.run_local_server(port=0)
                with open('token.json', 'w') as token:
                    token.write(creds.to_json())

            self.service = build('gmail', 'v1', credentials=creds)
            return True
        except Exception as e:
            print(f"Lỗi kết nối: {str(e)}")
            return False

    def get_emails(self, max_results=10):
        try:
            results = self.service.users().messages().list(
                userId='me',
                labelIds=['INBOX'],
                maxResults=max_results
            ).execute()
            return results.get('messages', [])
        except Exception as e:
            print(f"Lỗi khi lấy email: {str(e)}")
            return []

    def decode_email_content(self, payload):
        if 'parts' in payload:
            parts = []
            for part in payload['parts']:
                if part['mimeType'] in ['text/plain', 'text/html']:
                    if 'data' in part['body']:
                        text = base64.urlsafe_b64decode(
                            part['body']['data']).decode('utf-8', 'ignore')
                        if part['mimeType'] == 'text/html':
                            text = BeautifulSoup(
                                text, 'html.parser').get_text()
                        parts.append(text)
            return '\n'.join(parts)
        elif 'body' in payload and 'data' in payload['body']:
            text = base64.urlsafe_b64decode(
                payload['body']['data']).decode('utf-8', 'ignore')
            if payload['mimeType'] == 'text/html':
                return BeautifulSoup(text, 'html.parser').get_text()
            return text
        return ""

    def get_email_content(self, msg_id, summary_sentences=3):
        try:
            message = self.service.users().messages().get(
                userId='me',
                id=msg_id,
                format='full'
            ).execute()

            headers = message['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'].lower(
            ) == 'subject'), 'Không có tiêu đề')
            sender = next((h['value'] for h in headers if h['name'].lower(
            ) == 'from'), 'Không rõ người gửi')
            date = next((h['value']
                        for h in headers if h['name'].lower() == 'date'), '')

            content = self.decode_email_content(message['payload'])
            plain_text = BeautifulSoup(content, 'html.parser').get_text()
            summary = self.summarizer.summarize(plain_text, summary_sentences)

            return {
                'subject': subject,
                'from': sender,
                'date': date,
                'body': content,
                'summary': summary
            }
        except Exception as e:
            print(f"Lỗi khi đọc email: {str(e)}")
            return None


class EmailAnalyzerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Phân tích và Tóm tắt Email")
        self.root.geometry(Constants.WINDOW_SIZE)

        # Loading screen
        self.loading_frame = ttk.Frame(self.root)
        self.loading_frame.place(relx=0.5, rely=0.5, anchor="center")

        self.loading_label = ttk.Label(
            self.loading_frame,
            text="Đang khởi tạo chương trình...",
            font=(Constants.FONT_FAMILY, Constants.FONT_SIZE_NORMAL)
        )
        self.loading_label.pack(pady=10)

        self.loading_progress = ttk.Progressbar(
            self.loading_frame,
            mode='indeterminate',
            length=200
        )
        self.loading_progress.pack()
        self.loading_progress.start()

        # Initialize in separate thread
        self.init_thread = threading.Thread(target=self.async_init)
        self.init_thread.start()

    def async_init(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            futures.append(executor.submit(ResourceLoader.init_nltk))

            self.html_converter = html2text.HTML2Text()
            self.html_converter.ignore_links = False
            self.html_converter.body_width = 0
            self.html_converter.ignore_images = False
            self.html_converter.ignore_tables = False

            concurrent.futures.wait(futures)

        self.root.after(0, self.complete_init)

    def complete_init(self):
        self.loading_frame.destroy()
        self.style = ttk.Style()
        self.setup_styles()
        self.analyzer = GmailAnalyzer()
        self.setup_gui()

    def setup_styles(self):
        self.style.configure(
            'Accent.TButton',
            padding=(Constants.PADDING, Constants.SMALL_PADDING),
            font=(Constants.FONT_FAMILY, Constants.FONT_SIZE_BUTTON, 'bold'),
            background=Constants.COLORS['primary'],
            foreground=Constants.COLORS['text_button'],
            relief='raised',
            borderwidth=2
        )

        self.style.map('Accent.TButton',
                       background=[
                           ('active', Constants.COLORS['primary_light']),
                           ('disabled', Constants.COLORS['disabled'])
                       ],
                       foreground=[
                           ('active', Constants.COLORS['text_button']),
                           ('disabled', Constants.COLORS['text_button'])
                       ],
                       relief=[('pressed', 'sunken')]
                       )

        self.style.configure(
            'Custom.TFrame', background=Constants.COLORS['background'])
        self.style.configure('Custom.TLabel',
                             font=(Constants.FONT_FAMILY,
                                   Constants.FONT_SIZE_NORMAL),
                             background=Constants.COLORS['background'])
        self.style.configure('Custom.Treeview',
                             font=(Constants.FONT_FAMILY,
                                   Constants.FONT_SIZE_NORMAL),
                             rowheight=Constants.ROW_HEIGHT)
        self.style.configure('Custom.Treeview.Heading',
                             font=(Constants.FONT_FAMILY, Constants.FONT_SIZE_NORMAL, 'bold'))

    def setup_gui(self):
        main_frame = ttk.Frame(
            self.root, style='Custom.TFrame', padding=Constants.PADDING)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Phân tích và Tóm tắt Email",
            font=(Constants.FONT_FAMILY, Constants.FONT_SIZE_TITLE, 'bold'),
            style='Custom.TLabel'
        )
        title_label.pack(pady=(0, Constants.LARGE_PADDING))

        # Controls frame
        controls_frame = ttk.Frame(main_frame, style='Custom.TFrame')
        controls_frame.pack(fill=tk.X, pady=(0, Constants.PADDING))

        # Connect button
        self.connect_btn = ttk.Button(
            controls_frame,
            text="Kết nối Gmail",
            style='Accent.TButton',
            command=self.connect_gmail,
            width=Constants.BUTTON_WIDTH
        )
        self.connect_btn.pack(side=tk.LEFT, padx=Constants.PADDING)

        # Email count control
        ttk.Label(
            controls_frame,
            text="Số email:",
            style='Custom.TLabel'
        ).pack(side=tk.LEFT, padx=Constants.SMALL_PADDING)

        self.email_count = ttk.Spinbox(
            controls_frame,
            from_=Constants.MIN_EMAIL_COUNT,
            to=Constants.MAX_EMAIL_COUNT,
            width=Constants.SPINBOX_WIDTH,
            font=(Constants.FONT_FAMILY, Constants.FONT_SIZE_NORMAL)
        )
        self.email_count.set(Constants.DEFAULT_EMAIL_COUNT)
        self.email_count.pack(side=tk.LEFT, padx=Constants.SMALL_PADDING)

        # Summary sentences control
        ttk.Label(
            controls_frame,
            text="Số câu tóm tắt:",
            style='Custom.TLabel'
        ).pack(side=tk.LEFT, padx=Constants.SMALL_PADDING)

        self.summary_sentences = ttk.Spinbox(
            controls_frame,
            from_=Constants.MIN_SUMMARY_SENTENCES,
            to=Constants.MAX_SUMMARY_SENTENCES,
            width=Constants.SPINBOX_WIDTH,
            font=(Constants.FONT_FAMILY, Constants.FONT_SIZE_NORMAL)
        )
        self.summary_sentences.set(Constants.DEFAULT_SUMMARY_SENTENCES)
        self.summary_sentences.pack(side=tk.LEFT, padx=Constants.SMALL_PADDING)

        # Analyze button
        self.analyze_btn = ttk.Button(
            controls_frame,
            text="Phân tích",
            style='Accent.TButton',
            command=self.start_analysis,
            width=Constants.BUTTON_WIDTH
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=Constants.PADDING)
        self.analyze_btn['state'] = 'disabled'

        # Progress bar
        self.progress = ttk.Progressbar(
            controls_frame,
            mode='determinate',
            length=Constants.PROGRESS_LENGTH
        )
        self.progress.pack(side=tk.LEFT, fill=tk.X,
                           expand=True, padx=Constants.PADDING)

        # Content area
        content_paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        content_paned.pack(fill=tk.BOTH, expand=True, pady=Constants.PADDING)

        # Left side: Email list
        list_frame = ttk.LabelFrame(
            content_paned, text="Danh sách email", padding=Constants.SMALL_PADDING)
        content_paned.add(list_frame, weight=1)

        self.email_list = ttk.Treeview(
            list_frame,
            columns=('No', 'From', 'Subject', 'Date'),
            show='headings',
            style='Custom.Treeview'
        )

        self.email_list.heading('No', text='STT')
        self.email_list.heading('From', text='Từ')
        self.email_list.heading('Subject', text='Tiêu đề')
        self.email_list.heading('Date', text='Ngày')

        self.email_list.column(
            'No', width=Constants.COLUMN_NO_WIDTH, anchor='center')
        self.email_list.column('From', width=Constants.COLUMN_FROM_WIDTH)
        self.email_list.column('Subject', width=Constants.COLUMN_SUBJECT_WIDTH)
        self.email_list.column('Date', width=Constants.COLUMN_DATE_WIDTH)

        list_scroll = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self.email_list.yview)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.email_list.configure(yscrollcommand=list_scroll.set)
        self.email_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.email_list.bind('<<TreeviewSelect>>', self.on_select_email)

        # Right side: Content and Summary
        right_paned = ttk.PanedWindow(content_paned, orient=tk.VERTICAL)
        content_paned.add(right_paned, weight=2)

        # Email content frame
        content_frame = ttk.LabelFrame(
            right_paned, text="Nội dung email", padding=Constants.SMALL_PADDING)
        right_paned.add(content_frame, weight=2)

        self.email_info = ttk.Label(
            content_frame,
            text="",
            wraplength=Constants.WRAP_LENGTH,
            style='Custom.TLabel'
        )
        self.email_info.pack(
            fill=tk.X, padx=Constants.SMALL_PADDING, pady=Constants.SMALL_PADDING)

        self.content_text = scrolledtext.ScrolledText(
            content_frame,
            wrap=tk.WORD,
            font=(Constants.FONT_FAMILY, Constants.FONT_SIZE_NORMAL),
            padx=Constants.SMALL_PADDING,
            pady=Constants.SMALL_PADDING
        )
        self.content_text.pack(fill=tk.BOTH, expand=True)

        # Summary frame
        summary_frame = ttk.LabelFrame(
            right_paned, text="Tóm tắt email", padding=Constants.SMALL_PADDING)
        right_paned.add(summary_frame, weight=1)

        self.summary_text = scrolledtext.ScrolledText(
            summary_frame,
            wrap=tk.WORD,
            font=(Constants.FONT_FAMILY, Constants.FONT_SIZE_NORMAL),
            padx=Constants.SMALL_PADDING,
            pady=Constants.SMALL_PADDING
        )
        self.summary_text.pack(fill=tk.BOTH, expand=True)

        # Configure tags
        self.content_text.tag_configure('bold',
                                        font=(Constants.FONT_FAMILY, Constants.FONT_SIZE_NORMAL, 'bold'))
        self.content_text.tag_configure('italic',
                                        font=(Constants.FONT_FAMILY, Constants.FONT_SIZE_NORMAL, 'italic'))
        self.content_text.tag_configure('link',
                                        foreground=Constants.COLORS['link'],
                                        underline=True)

        # Context menu
        self.content_text_menu = tk.Menu(self.content_text, tearoff=0)
        self.content_text_menu.add_command(
            label="Sao chép", command=self.copy_text)
        self.content_text_menu.add_command(
            label="Chọn tất cả", command=self.select_all_text)
        self.content_text.bind("<Button-3>", self.show_context_menu)

        # Status bar
        self.status_var = tk.StringVar(value="Chưa kết nối")
        self.status_bar = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            style='Custom.TLabel',
            relief=tk.SUNKEN,
            padding=Constants.SMALL_PADDING
        )
        self.status_bar.pack(fill=tk.X, pady=(Constants.PADDING, 0))

    def format_html_content(self, content):
        try:
            text = self.html_converter.handle(content)
            text = html.unescape(text)
            text = re.sub(r'\n\s*\n', '\n\n', text)
            return text
        except Exception as e:
            print(f"Lỗi khi format HTML: {str(e)}")
            return content

    def apply_text_formatting(self):
        content = self.content_text.get('1.0', tk.END)

        for match in re.finditer(r'^#.*$', content, re.MULTILINE):
            start = f"1.0+{match.start()}c"
            end = f"1.0+{match.end()}c"
            self.content_text.tag_add('bold', start, end)

        for match in re.finditer(r'\*(.*?)\*', content):
            start = f"1.0+{match.start()}c"
            end = f"1.0+{match.end()}c"
            self.content_text.tag_add('italic', start, end)

        for match in re.finditer(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content):
            start = f"1.0+{match.start()}c"
            end = f"1.0+{match.end()}c"
            self.content_text.tag_add('link', start, end)

    def connect_gmail(self):
        self.status_var.set("Đang kết nối...")
        self.connect_btn['state'] = 'disabled'

        def connect_thread():
            if self.analyzer.connect():
                self.root.after(0, self.connection_success)
            else:
                self.root.after(0, self.connection_failed)

        threading.Thread(target=connect_thread).start()

    def connection_success(self):
        self.status_var.set("Đã kết nối thành công")
        self.analyze_btn['state'] = 'normal'
        messagebox.showinfo("Kết nối", "Kết nối Gmail thành công!")

    def connection_failed(self):
        self.status_var.set("Kết nối thất bại")
        self.connect_btn['state'] = 'normal'
        messagebox.showerror("Lỗi", "Không thể kết nối với Gmail!")

    def start_analysis(self):
        self.analyze_btn['state'] = 'disabled'
        self.email_list.delete(*self.email_list.get_children())
        self.content_text.delete('1.0', tk.END)
        self.summary_text.delete('1.0', tk.END)
        self.progress['value'] = 0
        count = int(self.email_count.get())

        def analyze_thread():
            messages = self.analyzer.get_emails(count)
            total = len(messages)

            for i, msg in enumerate(messages, 1):
                email_data = self.analyzer.get_email_content(
                    msg['id'],
                    int(self.summary_sentences.get())
                )
                if email_data:
                    self.root.after(0, self.add_email_to_list,
                                    email_data, msg['id'], i)
                progress = int((i / total) * 100)
                self.root.after(0, self.update_progress, progress)

            self.root.after(0, self.analysis_complete)

        threading.Thread(target=analyze_thread).start()

    def add_email_to_list(self, email_data, msg_id, index):
        self.email_list.insert('', 'end', iid=msg_id, values=(
            index,
            email_data['from'],
            email_data['subject'],
            email_data['date']
        ))

    def update_progress(self, value):
        self.progress['value'] = value
        self.status_var.set(f"Đang xử lý... {value}%")

    def analysis_complete(self):
        self.analyze_btn['state'] = 'normal'
        self.status_var.set("Phân tích hoàn tất")
        messagebox.showinfo("Hoàn thành", "Đã phân tích xong email!")

    def on_select_email(self, event):
        selection = self.email_list.selection()
        if selection:
            msg_id = selection[0]
            email_data = self.analyzer.get_email_content(
                msg_id,
                int(self.summary_sentences.get())
            )
            if email_data:
                info_text = f"""Từ: {email_data['from']}
Tiêu đề: {email_data['subject']}
Ngày: {email_data['date']}"""

                self.email_info.config(text=info_text)

                # Update content
                formatted_content = self.format_html_content(
                    email_data['body'])
                self.content_text.delete('1.0', tk.END)
                self.content_text.insert('1.0', formatted_content)
                self.apply_text_formatting()

                # Update summary
                self.summary_text.delete('1.0', tk.END)
                self.summary_text.insert('1.0', email_data['summary'])

    def copy_text(self):
        try:
            selected_text = self.content_text.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.root.clipboard_clear()
            self.root.clipboard_append(selected_text)
        except tk.TclError:
            pass

    def select_all_text(self):
        self.content_text.tag_add(tk.SEL, "1.0", tk.END)
        self.content_text.mark_set(tk.INSERT, "1.0")
        self.content_text.see(tk.INSERT)

    def show_context_menu(self, event):
        self.content_text_menu.tk_popup(event.x_root, event.y_root)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = EmailAnalyzerGUI()
    app.run()
