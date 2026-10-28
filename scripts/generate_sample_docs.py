"""Generate the fictional sample PDFs. Usage: python scripts/generate_sample_docs.py [en|id|all]"""
import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sample_content_en as en  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "samples"

NOTICE_ID = (
    "Dokumen ini adalah contoh fiktif untuk demonstrasi DocRAG. Perusahaan, "
    "kebijakan, dan angka di dalamnya tidak nyata."
)

TITLE = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=17, leading=22, spaceAfter=4)
SUBTITLE = ParagraphStyle("subtitle", fontName="Helvetica", fontSize=10.5, leading=14, spaceAfter=18)
BAB = ParagraphStyle("bab", fontName="Helvetica-Bold", fontSize=13, leading=18, spaceBefore=14, spaceAfter=6)
PASAL = ParagraphStyle("pasal", fontName="Helvetica-Bold", fontSize=11, leading=15, spaceBefore=10, spaceAfter=3)
BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=10.5, leading=15, spaceAfter=6)
SMALL = ParagraphStyle("small", fontName="Helvetica-Oblique", fontSize=9, leading=12, spaceBefore=16)


class NumberedCanvas(canvas.Canvas):
    """Adds a 'Page X of Y' footer, like real policy documents have."""

    footer = "Halaman {page} dari {total}"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            self.setFont("Helvetica", 9)
            self.drawRightString(A4[0] - 2 * cm, 1.2 * cm, self.footer.format(page=self._pageNumber, total=total))
            super().showPage()
        super().save()


# Each document: (file name, title, number, [("BAB ..." or None, [(heading, [paragraphs])])])
# A heading starting with "Lampiran" begins on a new page.
DOCS_ID = [
    (
        "Panduan_Karyawan_Contoh.pdf",
        "Panduan Karyawan",
        "Nomor PK-001/2026",
        [
            ("BAB I KETENTUAN UMUM", [
                ("Pasal 1 Tujuan", ["Panduan ini mengatur hak dan kewajiban karyawan PT Contoh Nusantara agar hubungan kerja berjalan tertib, adil, dan transparan."]),
                ("Pasal 2 Ruang Lingkup", ["Panduan ini berlaku bagi seluruh karyawan tetap dan karyawan kontrak di semua unit kerja. Ketentuan khusus bagi karyawan kontrak diatur dalam perjanjian kerja masing-masing."]),
            ]),
            ("BAB II JAM KERJA", [
                ("Pasal 3 Hari dan Jam Kerja", ["Hari kerja adalah Senin sampai Jumat. Jam kerja dimulai pukul 08.00 dan berakhir pukul 17.00, dengan waktu istirahat satu jam pada pukul 12.00.", "Karyawan dapat bekerja dari rumah paling banyak dua hari dalam seminggu dengan persetujuan atasan langsung."]),
                ("Pasal 4 Lembur", ["Lembur harus disetujui secara tertulis oleh atasan sebelum dilaksanakan. Kompensasi lembur diberikan berupa uang lembur atau pengganti waktu istirahat sesuai kesepakatan."]),
            ]),
            ("BAB III CUTI", [
                ("Pasal 5 Cuti Tahunan", ["Karyawan tetap berhak atas 12 hari cuti tahunan setelah menyelesaikan masa kerja satu tahun. Rincian dan tata cara pengajuan cuti diatur dalam Kebijakan Cuti."]),
                ("Pasal 6 Cuti Lainnya", ["Karyawan berhak atas cuti sakit, cuti melahirkan, dan cuti karena alasan penting sesuai ketentuan yang berlaku di perusahaan."]),
            ]),
            ("BAB IV DISIPLIN", [
                ("Pasal 7 Kewajiban Karyawan", ["Karyawan wajib menjaga kerahasiaan data perusahaan, menggunakan fasilitas kantor secara wajar, dan melaporkan benturan kepentingan kepada atasan."]),
                ("Pasal 8 Sanksi", ["Pelanggaran terhadap panduan ini dapat dikenai teguran lisan, teguran tertulis, atau tindakan lain sesuai tingkat pelanggaran, setelah karyawan diberi kesempatan menjelaskan."]),
            ]),
            (None, [
                ("Lampiran 1 Daftar Kontak Unit", ["Pertanyaan tentang panduan ini dapat disampaikan kepada bagian Sumber Daya Manusia melalui surel hr@contoh.example atau ekstensi 101 pada hari kerja."]),
            ]),
        ],
    ),
    (
        "Kebijakan_Cuti_Contoh.pdf",
        "Kebijakan Cuti",
        "Nomor KC-002/2026",
        [
            ("BAB I KETENTUAN UMUM", [
                ("Pasal 1 Definisi", ["Cuti adalah ketidakhadiran karyawan dari pekerjaan yang disetujui oleh perusahaan. Hari kerja adalah Senin sampai Jumat, tidak termasuk hari libur nasional."]),
            ]),
            ("BAB II JENIS CUTI", [
                ("Pasal 2 Cuti Tahunan", ["Karyawan tetap berhak atas 12 hari cuti tahunan setelah menyelesaikan masa kerja satu tahun.", "Cuti tahunan yang belum diambil dapat diakumulasikan paling banyak 6 hari untuk tahun berikutnya. Sisa cuti di atas batas tersebut dinyatakan hangus pada akhir bulan Maret."]),
                ("Pasal 3 Pengajuan Cuti", ["Pengajuan cuti tahunan dilakukan paling lambat 3 hari kerja sebelum tanggal cuti melalui atasan langsung, dengan mengisi formulir pada lampiran kebijakan ini.", "Atasan memberikan keputusan paling lambat 2 hari kerja setelah pengajuan diterima."]),
                ("Pasal 4 Cuti Sakit", ["Karyawan yang sakit satu sampai dua hari cukup memberi tahu atasan. Sakit lebih dari dua hari berturut-turut harus disertai surat keterangan dokter."]),
                ("Pasal 5 Cuti Melahirkan", ["Karyawati berhak atas cuti melahirkan selama 3 bulan dengan upah penuh. Suami berhak atas cuti pendampingan selama 5 hari kerja."]),
                ("Pasal 6 Cuti Karena Alasan Penting", ["Cuti karena alasan penting diberikan paling banyak 3 hari kerja untuk pernikahan karyawan, 2 hari kerja untuk pernikahan anak, dan 3 hari kerja untuk duka cita keluarga inti."]),
            ]),
            (None, [
                ("Lampiran 1 Formulir Pengajuan Cuti", ["Formulir memuat nama karyawan, unit kerja, jenis cuti, tanggal mulai dan berakhir, nama pengganti tugas, serta persetujuan atasan dan bagian Sumber Daya Manusia."]),
                ("Lampiran 2 Ringkasan Hak Cuti", ["Cuti tahunan 12 hari, cuti sakit sesuai surat dokter, cuti melahirkan 3 bulan, cuti pendampingan 5 hari kerja, dan cuti pernikahan 3 hari kerja."]),
            ]),
        ],
    ),
    (
        "Kebijakan_Perjalanan_Dinas_Contoh.pdf",
        "Kebijakan Perjalanan Dinas",
        "Nomor PD-003/2026",
        [
            ("BAB I KETENTUAN UMUM", [
                ("Pasal 1 Tujuan", ["Kebijakan ini mengatur persetujuan, biaya, dan pertanggungjawaban perjalanan dinas karyawan PT Contoh Nusantara."]),
                ("Pasal 2 Persetujuan Perjalanan", ["Perjalanan dinas harus disetujui oleh atasan dan bagian Keuangan paling lambat 5 hari kerja sebelum keberangkatan."]),
            ]),
            ("BAB II BIAYA PERJALANAN", [
                ("Pasal 3 Moda Transportasi", ["Perjalanan udara menggunakan kelas ekonomi. Perjalanan darat menggunakan kendaraan umum atau kendaraan perusahaan.", "Penggunaan taksi dibatasi untuk perpindahan dari dan ke bandara atau stasiun."]),
                ("Pasal 4 Uang Harian", ["Uang harian perjalanan dinas dalam negeri sebesar Rp450.000 per hari dan mencakup makan serta biaya kecil lainnya.", "Biaya penginapan ditanggung sesuai bukti pembayaran dengan batas Rp900.000 per malam."]),
                ("Pasal 5 Klaim Biaya", ["Klaim biaya perjalanan dinas diajukan paling lambat 7 hari kerja setelah karyawan kembali, dengan melampirkan bukti pembayaran asli. Klaim yang terlambat diproses pada periode pembayaran berikutnya."]),
            ]),
            (None, [
                ("Lampiran 1 Formulir Klaim Perjalanan Dinas", ["Formulir memuat tujuan perjalanan, tanggal berangkat dan kembali, rincian biaya per kategori, dan tanda tangan atasan."]),
            ]),
        ],
    ),
    (
        "Kebijakan_Keamanan_Informasi_Contoh.pdf",
        "Kebijakan Keamanan Informasi",
        "Nomor KI-004/2026",
        [
            ("BAB I KETENTUAN UMUM", [
                ("Pasal 1 Tujuan", ["Kebijakan ini melindungi data dan sistem informasi PT Contoh Nusantara dari akses tidak sah, kehilangan, dan penyalahgunaan."]),
            ]),
            ("BAB II PERLINDUNGAN AKUN DAN PERANGKAT", [
                ("Pasal 2 Kata Sandi", ["Kata sandi paling sedikit 12 karakter, memuat huruf besar, huruf kecil, angka, dan simbol, serta diganti paling lama setiap 90 hari.", "Kata sandi tidak boleh dibagikan kepada siapa pun atau dituliskan di tempat terbuka."]),
                ("Pasal 3 Perangkat", ["Laptop perusahaan wajib memakai enkripsi disk dan kunci layar otomatis setelah 5 menit tidak aktif. Penggunaan perangkat pribadi untuk mengakses data perusahaan memerlukan persetujuan tim keamanan informasi."]),
            ]),
            ("BAB III DATA DAN INSIDEN", [
                ("Pasal 4 Klasifikasi Data", ["Data diklasifikasikan menjadi publik, internal, dan rahasia. Data rahasia hanya boleh dikirim melalui kanal terenkripsi yang disetujui."]),
                ("Pasal 5 Pelaporan Insiden", ["Setiap dugaan insiden keamanan informasi wajib dilaporkan kepada tim keamanan informasi paling lambat 24 jam setelah diketahui."]),
            ]),
            (None, [
                ("Lampiran 1 Daftar Kontak Tanggap Insiden", ["Tim keamanan informasi dapat dihubungi melalui surel security@contoh.example atau ekstensi 199 setiap hari kerja pukul 08.00 sampai 17.00."]),
            ]),
        ],
    ),
    (
        "Prosedur_Pengadaan_Contoh.pdf",
        "Prosedur Pengadaan Barang dan Jasa",
        "Nomor PP-005/2026",
        [
            ("BAB I KETENTUAN UMUM", [
                ("Pasal 1 Tujuan", ["Prosedur ini memastikan pengadaan barang dan jasa PT Contoh Nusantara dilakukan secara wajar, terdokumentasi, dan bebas dari benturan kepentingan."]),
            ]),
            ("BAB II METODE DAN PERSETUJUAN", [
                ("Pasal 2 Batas Nilai dan Metode", ["Pengadaan dengan nilai sampai dengan Rp10.000.000 dapat dilakukan melalui pembelian langsung.", "Pengadaan dengan nilai di atas Rp10.000.000 sampai Rp100.000.000 wajib membandingkan paling sedikit 3 penawaran.", "Pengadaan dengan nilai di atas Rp100.000.000 dilakukan melalui tender terbatas atau tender terbuka."]),
                ("Pasal 3 Persetujuan", ["Permintaan pengadaan harus disetujui kepala unit dan bagian Keuangan. Pengadaan di atas Rp100.000.000 juga memerlukan persetujuan direktur."]),
            ]),
            ("BAB III PENYEDIA", [
                ("Pasal 4 Evaluasi Penyedia", ["Penyedia dievaluasi berdasarkan harga, kualitas, ketepatan waktu, dan kepatuhan administrasi. Penyedia dengan catatan keterlambatan berulang dapat dikeluarkan dari daftar penyedia."]),
                ("Pasal 5 Konflik Kepentingan", ["Karyawan yang memiliki hubungan keluarga atau kepentingan finansial dengan penyedia wajib menyatakan hal tersebut dan tidak ikut dalam proses penilaian."]),
            ]),
            (None, [
                ("Lampiran 1 Daftar Periksa Dokumen Pengadaan", ["Dokumen yang harus lengkap: permintaan pengadaan, perbandingan penawaran bila diwajibkan, persetujuan sesuai batas nilai, kontrak atau pesanan pembelian, dan berita acara serah terima."]),
                ("Lampiran 2 Format Berita Acara Serah Terima", ["Berita acara memuat tanggal serah terima, rincian barang atau jasa, kondisi saat diterima, dan tanda tangan penerima serta penyedia."]),
            ]),
        ],
    ),
]


LANGUAGES = {
    "id": {"docs": DOCS_ID, "notice": NOTICE_ID, "company": "PT Contoh Nusantara", "footer": "Halaman {page} dari {total}", "appendix": "Lampiran"},
    "en": {"docs": en.DOCS, "notice": en.NOTICE, "company": en.COMPANY, "footer": en.FOOTER, "appendix": "Appendix"},
}


def build(out_dir, lang, file_name, title, number, groups):
    cfg = LANGUAGES[lang]
    canvas_cls = type("Canvas", (NumberedCanvas,), {"footer": cfg["footer"]})
    story = [Paragraph(title, TITLE), Paragraph(f"{cfg['company']} | {number}", SUBTITLE)]
    for bab, entries in groups:
        if bab:
            story.append(Paragraph(bab, BAB))
        for heading, paragraphs in entries:
            is_appendix = heading.startswith(cfg["appendix"])
            if is_appendix:
                story.append(PageBreak())
            story.append(Paragraph(heading, BAB if is_appendix else PASAL))
            for text in paragraphs:
                story.append(Paragraph(text, BODY))
    story.append(Spacer(1, 12))
    story.append(Paragraph(cfg["notice"], SMALL))
    doc = SimpleDocTemplate(
        str(out_dir / file_name), pagesize=A4, title=title, author="DocRAG samples",
        leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2.2 * cm, bottomMargin=2.2 * cm,
    )
    doc.build(story, canvasmaker=canvas_cls)


if __name__ == "__main__":
    choice = sys.argv[1] if len(sys.argv) > 1 else "all"
    for lang in (LANGUAGES if choice == "all" else [choice]):
        out_dir = OUT / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        for spec in LANGUAGES[lang]["docs"]:
            build(out_dir, lang, *spec)
            print("wrote", out_dir / spec[0])
