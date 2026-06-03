from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

# Create PDF
pdf_file = "QKD_Simulator_Survey.pdf"
doc = SimpleDocTemplate(pdf_file, pagesize=letter, rightMargin=0.5*inch, leftMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)

# Container for PDF elements
elements = []

# Define styles
styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=18,
    textColor=colors.HexColor('#1f4788'),
    spaceAfter=12,
    alignment=TA_CENTER,
    fontName='Helvetica-Bold'
)

heading_style = ParagraphStyle(
    'CustomHeading',
    parent=styles['Heading2'],
    fontSize=14,
    textColor=colors.HexColor('#2d5aa8'),
    spaceAfter=10,
    spaceBefore=10,
    fontName='Helvetica-Bold'
)

question_style = ParagraphStyle(
    'Question',
    parent=styles['Normal'],
    fontSize=11,
    textColor=colors.black,
    spaceAfter=6,
    fontName='Helvetica-Bold'
)

option_style = ParagraphStyle(
    'Option',
    parent=styles['Normal'],
    fontSize=10,
    textColor=colors.black,
    spaceAfter=4,
    leftIndent=20,
    fontName='Helvetica'
)

answer_style = ParagraphStyle(
    'Answer',
    parent=styles['Normal'],
    fontSize=9,
    textColor=colors.HexColor('#00aa00'),
    spaceAfter=8,
    leftIndent=20,
    fontName='Helvetica-Oblique'
)

# Title
elements.append(Paragraph("QKD Simulator Survey", title_style))
elements.append(Paragraph("Quantum Key Distribution as an Educational Tool", styles['Heading3']))
elements.append(Spacer(1, 0.3*inch))

# Pre-Simulation Quiz
elements.append(Paragraph("PRE-SIMULATION QUIZ", heading_style))
elements.append(Paragraph("Knowledge Assessment", styles['Normal']))
elements.append(Spacer(1, 0.1*inch))

pre_questions = [
    {
        "q": "Question 1: What is the primary purpose of Quantum Key Distribution (QKD)?",
        "options": [
            "A) To encrypt data faster than classical methods",
            "B) To distribute cryptographic keys securely between two parties",
            "C) To detect quantum particles in real-time",
            "D) To replace all classical encryption methods"
        ],
        "answer": "Correct Answer: B"
    },
    {
        "q": "Question 2: What is the BB84 protocol?",
        "options": [
            "A) A quantum computing algorithm for factoring large numbers",
            "B) A quantum key distribution protocol using photon polarization",
            "C) A method to amplify quantum signals",
            "D) A classical encryption technique used in 1984"
        ],
        "answer": "Correct Answer: B"
    },
    {
        "q": "Question 3: In the BB84 protocol, what are the two types of bases used?",
        "options": [
            "A) Linear and Angular bases",
            "B) Rectilinear and Diagonal bases",
            "C) Horizontal and Vertical bases",
            "D) Real and Imaginary bases"
        ],
        "answer": "Correct Answer: B"
    },
    {
        "q": "Question 4: How does an eavesdropper affect quantum key distribution?",
        "options": [
            "A) They cannot be detected as quantum states remain unchanged",
            "B) Their measurement collapses the quantum state, introducing detectable errors",
            "C) They only affect the key 50% of the time",
            "D) Eavesdroppers are impossible to identify in QKD systems"
        ],
        "answer": "Correct Answer: B"
    },
    {
        "q": "Question 5: What is the quantum uncertainty principle's relevance to QKD security?",
        "options": [
            "A) It proves that faster-than-light communication is possible",
            "B) It guarantees that measuring a quantum state changes it, enabling eavesdropper detection",
            "C) It limits the number of bits that can be transmitted",
            "D) It has no practical application in cryptography"
        ],
        "answer": "Correct Answer: B"
    }
]

for q_data in pre_questions:
    elements.append(Paragraph(q_data["q"], question_style))
    for option in q_data["options"]:
        elements.append(Paragraph(option, option_style))
    elements.append(Paragraph(q_data["answer"], answer_style))
    elements.append(Spacer(1, 0.1*inch))

# Page break
elements.append(PageBreak())

# Post-Simulation Quiz
elements.append(Paragraph("POST-SIMULATION QUIZ", heading_style))
elements.append(Paragraph("Application & Understanding", styles['Normal']))
elements.append(Spacer(1, 0.1*inch))

post_questions = [
    {
        "q": "Question 1: After running the simulator, which basis mismatch rate did you observe?",
        "options": [
            "A) 0% - all bases match perfectly",
            "B) Approximately 25% - matching occurs when both use the same basis",
            "C) Approximately 50% - bases mismatch randomly",
            "D) 100% - bases never match"
        ],
        "answer": "Correct Answer: C"
    },
    {
        "q": "Question 2: In the simulator, how does the presence of an eavesdropper affect the quantum bit error rate (QBER)?",
        "options": [
            "A) QBER remains at 0% regardless of eavesdropping",
            "B) QBER increases to approximately 50% when Eve eavesdrops",
            "C) QBER increases to approximately 25% when Eve eavesdrops",
            "D) QBER cannot be measured in simulations"
        ],
        "answer": "Correct Answer: C"
    },
    {
        "q": "Question 3: What was the final sifted key length after eliminating basis mismatches in your simulation?",
        "options": [
            "A) The same as the original quantum bits sent",
            "B) Approximately 25% of the original quantum bits",
            "C) Approximately 50% of the original quantum bits",
            "D) Zero - all bits are discarded"
        ],
        "answer": "Correct Answer: C"
    },
    {
        "q": "Question 4: How does the simulator demonstrate the advantage of QKD over classical key exchange?",
        "options": [
            "A) It transmits keys faster than light",
            "B) It allows eavesdropper detection while classical methods do not",
            "C) It uses fewer computational resources",
            "D) It requires no quantum hardware"
        ],
        "answer": "Correct Answer: B"
    },
    {
        "q": "Question 5: Based on the simulator's noise/error analysis, what would be a typical action if you detected unusual QBER levels?",
        "options": [
            "A) Accept the key without further verification",
            "B) Abort the key exchange and investigate for potential eavesdropping",
            "C) Repeat the transmission to average out errors",
            "D) Use the noisy key anyway for encryption"
        ],
        "answer": "Correct Answer: B"
    }
]

for q_data in post_questions:
    elements.append(Paragraph(q_data["q"], question_style))
    for option in q_data["options"]:
        elements.append(Paragraph(option, option_style))
    elements.append(Paragraph(q_data["answer"], answer_style))
    elements.append(Spacer(1, 0.1*inch))

# Build PDF
doc.build(elements)
print(f"PDF created successfully: {pdf_file}")
