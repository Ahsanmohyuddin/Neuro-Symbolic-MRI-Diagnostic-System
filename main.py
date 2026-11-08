# ════════════════════════════════════════════════════════════════
# main.py  –  NeuroScan Alzheimer Classifier  (Full Version)
#
# Routes:
#   GET  /           → health check
#   POST /predict    → MRI image → prediction + Grad-CAM + full data
#   POST /survey     → survey answers → score + evaluation
#   POST /generate-pdf → all data → downloadable PDF report
# ════════════════════════════════════════════════════════════════

import io, os, base64, json
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from PIL import Image
from torchvision import models, transforms

# ReportLab for PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ── App ─────────────────────────────────────────────────────────
app = FastAPI(title="NeuroScan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ════════════════════════════════════════════════════════════════
# DATA — All class info, descriptions, questions, doctors
# ════════════════════════════════════════════════════════════════

# Class names must match ImageFolder alphabetical order from training
CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

# ── Severity score (0–100) per class ────────────────────────────
SEVERITY = {
    "NonDemented":      8,
    "VeryMildDemented": 28,
    "MildDemented":     58,
    "ModerateDemented": 82,
}

# ── Severity zone label + color ──────────────────────────────────
SEVERITY_ZONE = {
    "NonDemented":      {"zone": "Safe Zone",       "color": "#22c55e", "label": "Safe"},
    "VeryMildDemented": {"zone": "Mild Risk",        "color": "#facc15", "label": "Early Stage"},
    "MildDemented":     {"zone": "Moderate Risk",    "color": "#f97316", "label": "Concerning"},
    "ModerateDemented": {"zone": "High Risk",        "color": "#ef4444", "label": "Critical"},
}

# ── Rich descriptions per class ──────────────────────────────────
DESCRIPTIONS = {
    "NonDemented": {
        "ui_color":   "#22c55e",
        "headline":   "Great News — No Signs Detected",
        "status":     "safe",
        "definition": "Your MRI scan shows no signs of Alzheimer's-related brain changes. The brain structure appears healthy and cognitive function is preserved.",
        "causes":     ["Genetic predisposition (if family history exists)", "Lifestyle factors like poor diet, inactivity", "Cardiovascular conditions if unmanaged", "Chronic stress and poor sleep over time"],
        "symptoms":   ["No current cognitive symptoms detected", "Normal memory and reasoning ability", "Occasional forgetfulness within normal range for age"],
        "precautions":["Get annual brain health check-ups", "Monitor blood pressure and cholesterol", "Avoid smoking and excessive alcohol", "Stay socially and mentally active"],
        "diet":       ["Mediterranean diet rich in leafy greens and fish", "Blueberries, walnuts, and olive oil for brain health", "Reduce processed food, sugar, and trans fats", "Stay well-hydrated — minimum 8 glasses of water daily"],
        "severity_explanation": "Your brain scan is in the safe zone. This is excellent news. However, maintaining a brain-healthy lifestyle is key to staying this way.",
        "good_news":  "Your scan indicates a healthy brain. Keep up the great work with your lifestyle choices!",
        "preventive_care": ["Daily 30-minute aerobic exercise", "Mental exercises: reading, puzzles, learning new skills", "Quality sleep of 7–9 hours per night", "Regular social interaction and stress management"],
        "recommendation": "Continue healthy lifestyle. Schedule annual neurological check-up as a preventive measure.",
    },
    "VeryMildDemented": {
        "ui_color":   "#facc15",
        "headline":   "Early Stage Detected — Act Now",
        "status":     "early",
        "definition": "The scan shows very mild signs of cognitive change. This is the earliest detectable stage of Alzheimer's. Symptoms are minimal and often only noticed by the patient.",
        "causes":     ["Accumulation of amyloid-beta plaques beginning", "Age-related neural changes", "Genetic factors (APOE-e4 gene)", "Chronic lifestyle factors: stress, poor sleep, inactivity"],
        "symptoms":   ["Occasional forgetfulness — names, recent events", "Slight difficulty with complex tasks", "Mild word-finding difficulties", "Generally functioning independently"],
        "precautions":["Consult a neurologist for proper evaluation", "Start a structured daily routine", "Avoid situations requiring quick complex decisions under stress", "Use memory aids: calendars, notebooks, reminders"],
        "diet":       ["Increase omega-3 intake: salmon, flaxseed, walnuts", "Turmeric (curcumin) — powerful anti-inflammatory for brain", "Green tea and dark chocolate for antioxidants", "Cut back on red meat and dairy fat"],
        "severity_explanation": "This is a mild risk stage. The changes are real but manageable. Early intervention is the single most impactful step you can take right now.",
        "good_news":  "Early detection is powerful. With the right steps now, progression can be significantly slowed.",
        "preventive_care": ["Start cognitive training programs", "Join a memory support group", "Daily journaling to track cognitive changes", "Consult a doctor about preventive medications"],
        "recommendation": "Schedule a consultation with a neurologist within the next 2–4 weeks. Early action is critical.",
    },
    "MildDemented": {
        "ui_color":   "#f97316",
        "headline":   "Mild Dementia Detected — Seek Medical Help",
        "status":     "warning",
        "definition": "The scan indicates mild Alzheimer's dementia. Cognitive decline is noticeable and is beginning to affect daily activities. Medical intervention and lifestyle modification are urgently recommended.",
        "causes":     ["Progressive amyloid plaque and tau tangle buildup", "Significant neuronal damage in hippocampus and cortex", "Genetic and environmental risk factors combined", "Prolonged unmanaged cardiovascular or metabolic conditions"],
        "symptoms":   ["Frequent forgetfulness affecting daily life", "Difficulty managing finances, medication, appointments", "Getting lost in familiar places", "Personality or mood changes", "Trouble following conversations"],
        "precautions":["Immediate neurologist consultation required", "Set up medication reminders and supervision", "Remove home safety hazards", "Do not drive without medical clearance", "Appoint a trusted family member as health coordinator"],
        "diet":       ["Strict Mediterranean diet — clinically proven to slow decline", "Avoid alcohol completely", "High antioxidant foods: berries, broccoli, spinach", "Small frequent meals to maintain blood sugar stability"],
        "severity_explanation": "This stage requires active medical management. Without intervention, decline may accelerate. With proper care, quality of life can be significantly maintained.",
        "warning": "This stage requires medical attention. Please contact a neurologist immediately.",
        "lifestyle_guidance": ["Establish a strict daily schedule", "Engage caregiver support", "Light daily exercise tailored to capability", "Memory therapy and cognitive stimulation activities"],
        "recommendation": "Urgent referral to neurologist and psychiatrist. Family support and professional care planning should begin immediately.",
    },
    "ModerateDemented": {
        "ui_color":   "#ef4444",
        "headline":   "Moderate Dementia — Urgent Care Required",
        "status":     "critical",
        "definition": "The scan shows significant Alzheimer's-related brain changes consistent with moderate dementia. Substantial cognitive and functional decline is present. Full-time care and specialist management are essential.",
        "causes":     ["Advanced amyloid and tau pathology throughout the brain", "Widespread neuronal death affecting multiple brain regions", "Genetic, environmental, and metabolic factors", "Disease progression from earlier untreated stages"],
        "symptoms":   ["Severe memory loss — may not recognize family", "Inability to perform daily tasks independently", "Significant personality and behavioral changes", "Wandering and disorientation", "Difficulty with language and communication"],
        "precautions":["24/7 supervised care required — home or facility", "Install home safety measures: locks, alarms", "Establish power of attorney for medical decisions", "Enroll in a specialist memory care program", "Regular psychiatric evaluation for behavioral management"],
        "diet":       ["Soft, easy-to-swallow foods due to swallowing difficulties", "High-nutrition meals with close monitoring", "Adequate hydration — risk of dehydration is high", "Vitamin D, B12, and folate supplementation as prescribed"],
        "severity_explanation": "This is a high-risk stage requiring full medical and caregiver support. The focus shifts from prevention to quality of life, dignity, and comfort management.",
        "warning": "URGENT: Immediate specialist medical care is required. Do not delay consultation.",
        "lifestyle_guidance": ["Professional memory care facility or home care nurse", "Music therapy and sensory stimulation", "Structured safe environment", "Family counselling and caregiver support groups"],
        "recommendation": "Immediate referral to a memory specialist and neuropsychiatrist. Evaluate memory care facility options. Begin advanced care planning.",
    },
}

# ── Survey questions (10 per class, type: 'risk' or 'protective') ─
# risk=Yes is bad, protective=Yes is good (used for scoring)
SURVEY_QUESTIONS = {
    "NonDemented": [
        {"id": 1, "text": "Do you experience occasional forgetfulness (e.g. misplacing keys)?",        "type": "risk"},
        {"id": 2, "text": "Do you exercise aerobically at least 3 times per week?",                    "type": "protective"},
        {"id": 3, "text": "Do you follow a balanced, nutritious diet daily?",                          "type": "protective"},
        {"id": 4, "text": "Do you get 7–9 hours of quality sleep each night?",                        "type": "protective"},
        {"id": 5, "text": "Do you engage in mentally stimulating activities (reading, puzzles)?",       "type": "protective"},
        {"id": 6, "text": "Do you have a family history of Alzheimer's disease?",                      "type": "risk"},
        {"id": 7, "text": "Do you feel unusually stressed or anxious most days?",                      "type": "risk"},
        {"id": 8, "text": "Do you have regular medical or neurological check-ups?",                    "type": "protective"},
        {"id": 9, "text": "Do you maintain an active social life?",                                    "type": "protective"},
        {"id": 10,"text": "Do you consume alcohol frequently (more than 2 drinks/day)?",               "type": "risk"},
    ],
    "VeryMildDemented": [
        {"id": 1, "text": "Do you sometimes forget where you placed common objects?",                   "type": "risk"},
        {"id": 2, "text": "Do you have difficulty recalling names of familiar people?",                 "type": "risk"},
        {"id": 3, "text": "Do you need reminders for medications or appointments?",                    "type": "risk"},
        {"id": 4, "text": "Have family members noticed changes in your memory?",                       "type": "risk"},
        {"id": 5, "text": "Do you feel frustrated when trying to remember things?",                    "type": "risk"},
        {"id": 6, "text": "Are you currently doing any cognitive training or brain exercises?",        "type": "protective"},
        {"id": 7, "text": "Do you maintain a regular daily routine?",                                  "type": "protective"},
        {"id": 8, "text": "Do you have any difficulty finding the right words in conversation?",       "type": "risk"},
        {"id": 9, "text": "Are you managing your finances independently without errors?",              "type": "protective"},
        {"id": 10,"text": "Are you currently seeing a doctor or specialist for memory concerns?",      "type": "protective"},
    ],
    "MildDemented": [
        {"id": 1, "text": "Do you frequently forget important dates or recent events?",                 "type": "risk"},
        {"id": 2, "text": "Do you need help with daily activities (cooking, cleaning, managing bills)?","type": "risk"},
        {"id": 3, "text": "Do you sometimes fail to recognise familiar faces?",                        "type": "risk"},
        {"id": 4, "text": "Do you experience significant mood or personality changes?",                "type": "risk"},
        {"id": 5, "text": "Do you get disoriented about the current date or year?",                   "type": "risk"},
        {"id": 6, "text": "Are you under regular medical supervision for cognitive decline?",          "type": "protective"},
        {"id": 7, "text": "Do you have a caregiver or support person helping you daily?",              "type": "protective"},
        {"id": 8, "text": "Do you participate in any structured cognitive therapy?",                   "type": "protective"},
        {"id": 9, "text": "Do you have trouble following conversations in group settings?",            "type": "risk"},
        {"id": 10,"text": "Do you engage in light daily physical activity?",                           "type": "protective"},
    ],
    "ModerateDemented": [
        {"id": 1, "text": "Do you sometimes fail to recognise immediate family members?",              "type": "risk"},
        {"id": 2, "text": "Do you require full assistance for personal care (dressing, bathing)?",    "type": "risk"},
        {"id": 3, "text": "Do you experience episodes of agitation, aggression, or wandering?",       "type": "risk"},
        {"id": 4, "text": "Do you have a full-time caregiver or live in a care facility?",             "type": "protective"},
        {"id": 5, "text": "Are you receiving regular psychiatric evaluation for behavioural symptoms?","type": "protective"},
        {"id": 6, "text": "Do you experience difficulty swallowing food or drinks?",                  "type": "risk"},
        {"id": 7, "text": "Do you sleep irregularly (sleeping during day, awake at night)?",          "type": "risk"},
        {"id": 8, "text": "Is an advanced care or end-of-life plan in place?",                        "type": "protective"},
        {"id": 9, "text": "Do you participate in music, art, or sensory stimulation therapy?",        "type": "protective"},
        {"id": 10,"text": "Does the family receive regular guidance from a dementia specialist?",      "type": "protective"},
    ],
}

# ── Doctor recommendations per stage ──────────────────────────────
DOCTORS = {
    "NonDemented": {
        "urgency": "Routine — annual check recommended",
        "specialist_types": ["General Physician", "Neurologist (annual screening)"],
        "message": "No immediate specialist visit required. An annual brain health check with your GP is sufficient.",
        "clinics": [
            {"name": "Shaukat Khanum Memorial Hospital", "city": "Lahore", "dept": "Neurology", "url": "https://www.shaukatkhanum.org.pk"},
            {"name": "Services Hospital Lahore", "city": "Lahore", "dept": "Neurology & Psychiatry", "url": "https://www.services-hospital.gop.pk"},
        ],
        "platforms": [
            {"name": "Marham.pk", "desc": "Find & book neurologists online", "url": "https://www.marham.pk/doctors/neurologist"},
            {"name": "Oladoc.com", "desc": "Online medical consultations",   "url": "https://oladoc.com/pakistan/lahore/neurologist"},
        ],
    },
    "VeryMildDemented": {
        "urgency": "Soon — within 2–4 weeks",
        "specialist_types": ["Neurologist", "Geriatrician", "Memory Specialist"],
        "message": "Early consultation is key. A neurologist can run formal cognitive assessments and recommend early interventions.",
        "clinics": [
            {"name": "PKLI & Research Centre",         "city": "Lahore", "dept": "Neurology",          "url": "https://www.pkli.org.pk"},
            {"name": "Jinnah Hospital Lahore",         "city": "Lahore", "dept": "Neuropsychiatry",    "url": "https://www.jhl.gop.pk"},
            {"name": "Ittefaq Hospital",               "city": "Lahore", "dept": "Neurology",          "url": "https://www.ittefaqhospital.com"},
        ],
        "platforms": [
            {"name": "Marham.pk", "desc": "Book a neurologist in Lahore", "url": "https://www.marham.pk/doctors/neurologist/lahore"},
            {"name": "Oladoc.com", "desc": "Memory specialist consultations", "url": "https://oladoc.com/pakistan/lahore/neurologist"},
        ],
    },
    "MildDemented": {
        "urgency": "Urgent — within 1 week",
        "specialist_types": ["Neurologist", "Psychiatrist", "Memory Specialist", "Geriatrician"],
        "message": "Immediate specialist care is needed. A neurologist will guide medication and a psychiatrist can manage behavioral symptoms.",
        "clinics": [
            {"name": "PKLI & Research Centre",         "city": "Lahore", "dept": "Neurology",                "url": "https://www.pkli.org.pk"},
            {"name": "Mayo Hospital Lahore",           "city": "Lahore", "dept": "Psychiatry & Neurology",   "url": "https://www.mayo-hospital.gop.pk"},
            {"name": "Services Hospital Lahore",       "city": "Lahore", "dept": "Neurology & Memory Clinic","url": "https://www.services-hospital.gop.pk"},
        ],
        "platforms": [
            {"name": "Marham.pk",  "desc": "Urgent neurologist booking — Lahore", "url": "https://www.marham.pk/doctors/neurologist/lahore"},
            {"name": "Oladoc.com", "desc": "Online psychiatrist & neurologist",   "url": "https://oladoc.com/pakistan/lahore/neurologist"},
        ],
    },
    "ModerateDemented": {
        "urgency": "URGENT — Immediate specialist care required",
        "specialist_types": ["Neurologist (Senior)", "Neuropsychiatrist", "Memory Care Specialist", "Palliative Care Physician"],
        "message": "Immediate specialist intervention is critical. Please arrange an emergency appointment. Full-time care planning and specialist management should begin today.",
        "clinics": [
            {"name": "PKLI & Research Centre",         "city": "Lahore", "dept": "Senior Neurology",             "url": "https://www.pkli.org.pk"},
            {"name": "Shaukat Khanum Memorial",        "city": "Lahore", "dept": "Neurology & Palliative Care",  "url": "https://www.shaukatkhanum.org.pk"},
            {"name": "Mayo Hospital Lahore",           "city": "Lahore", "dept": "Memory Care & Neuropsychiatry","url": "https://www.mayo-hospital.gop.pk"},
        ],
        "platforms": [
            {"name": "Marham.pk",  "desc": "Emergency neurologist booking — Lahore", "url": "https://www.marham.pk/doctors/neurologist/lahore"},
            {"name": "Oladoc.com", "desc": "Senior neurologist & psychiatrist",      "url": "https://oladoc.com/pakistan/lahore/neurologist"},
        ],
    },
}

# ════════════════════════════════════════════════════════════════
# MODEL SETUP
# ════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

# Same transform used during training (validation/test)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def load_model():
    global model
    path = os.path.join(os.path.dirname(__file__), "maxvit_best_model.pth")
    if not os.path.exists(path):
        print(f"⚠️  Model not found at: {path}")
        return
    net = models.convnext_small(pretrained=False)
    net.classifier[2] = nn.Linear(net.classifier[2].in_features, len(CLASS_NAMES))
    net.load_state_dict(torch.load(path, map_location=device))
    net.to(device)
    net.eval()
    model = net
    print(f"✅ Model loaded on {device}")

load_model()

# ════════════════════════════════════════════════════════════════
# GRAD-CAM
# ════════════════════════════════════════════════════════════════

def get_gradcam(image_tensor, target_class):
    """Generate Grad-CAM heatmap overlaid on MRI. Returns base64 PNG."""
    conv_output, gradients = None, None

    def fwd_hook(module, inp, out):
        nonlocal conv_output
        conv_output = out

    def bwd_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    layer = dict([*model.named_modules()])["features.7"]
    fh = layer.register_forward_hook(fwd_hook)
    bh = layer.register_backward_hook(bwd_hook)

    output = model(image_tensor)
    model.zero_grad()
    output[0, target_class].backward()
    fh.remove()
    bh.remove()

    # Weight activations by gradient mean → heatmap
    pooled = torch.mean(gradients, dim=[0, 2, 3])
    act = conv_output[0]
    for i in range(act.shape[0]):
        act[i] *= pooled[i]

    heatmap = torch.mean(act, dim=0).cpu().detach().numpy()
    heatmap  = np.maximum(heatmap, 0)
    heatmap  = heatmap / (heatmap.max() + 1e-8)

    # Reconstruct original image from normalised tensor
    img_np = image_tensor[0].cpu().detach().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    img_np = (img_np * 0.5 + 0.5) * 255
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    # Resize + colormap + blend
    hm = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    hm_color = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    blended  = cv2.addWeighted(img_np, 0.6, hm_color, 0.4, 0)

    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# ════════════════════════════════════════════════════════════════
# PYDANTIC REQUEST MODELS
# ════════════════════════════════════════════════════════════════

class SurveyRequest(BaseModel):
    predicted_class: str
    answers: List[int]     # 0=No, 1=Sometimes, 2=Yes per question

class PDFRequest(BaseModel):
    predicted_class:    str
    confidence:         float
    severity:           int
    survey_answers:     List[int]
    survey_questions:   List[str]
    survey_evaluation:  str
    survey_score:       int
    patient_name:       str = "Anonymous Patient"

# ════════════════════════════════════════════════════════════════
# PDF GENERATION
# ════════════════════════════════════════════════════════════════

def build_pdf(req: PDFRequest) -> bytes:
    """Build a professional PDF report using ReportLab and return bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=1.8*cm, bottomMargin=1.8*cm,
                            leftMargin=2*cm, rightMargin=2*cm)

    # ── Styles ────────────────────────────────────────────────────
    base   = getSampleStyleSheet()
    title  = ParagraphStyle("title",  parent=base["Normal"], fontSize=20, leading=26,
                             alignment=TA_CENTER, textColor=colors.HexColor("#1e3a5f"), spaceAfter=6)
    sub    = ParagraphStyle("sub",    parent=base["Normal"], fontSize=11, leading=15,
                             alignment=TA_CENTER, textColor=colors.HexColor("#4b6a9b"), spaceAfter=12)
    h1s    = ParagraphStyle("h1s",    parent=base["Normal"], fontSize=14, leading=18,
                             textColor=colors.HexColor("#1e3a5f"), spaceBefore=14, spaceAfter=5,
                             fontName="Helvetica-Bold")
    body   = ParagraphStyle("body",   parent=base["Normal"], fontSize=10, leading=15,
                             textColor=colors.HexColor("#1f2937"), spaceAfter=5)
    small  = ParagraphStyle("small",  parent=base["Normal"], fontSize=8,  leading=12,
                             textColor=colors.HexColor("#6b7280"), spaceAfter=3)
    warn   = ParagraphStyle("warn",   parent=base["Normal"], fontSize=11, leading=15,
                             textColor=colors.HexColor("#b91c1c"), spaceBefore=8, spaceAfter=8,
                             fontName="Helvetica-Bold")

    cls        = req.predicted_class
    desc       = DESCRIPTIONS.get(cls, {})
    sev_zone   = SEVERITY_ZONE.get(cls, {})
    doctors    = DOCTORS.get(cls, {})
    questions  = SURVEY_QUESTIONS.get(cls, [])

    clr_hex = SEVERITY_ZONE.get(cls, {}).get("color", "#3b82f6")
    clr     = colors.HexColor(clr_hex)

    # ── Build story ───────────────────────────────────────────────
    story = []

    # Header
    story.append(Paragraph("🧠  NeuroScan Diagnostic Report", title))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%d %B %Y, %I:%M %p')}", sub))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1e3a5f")))
    story.append(Spacer(1, 10))

    # Patient info
    info_data = [
        ["Patient Name:", req.patient_name, "Report Date:", datetime.now().strftime("%d/%m/%Y")],
        ["Diagnosis:",    desc.get("headline", cls), "Severity:", f"{req.severity} / 100"],
    ]
    t = Table(info_data, colWidths=[4*cm, 7*cm, 4*cm, 4*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#dbeafe")),
        ("BACKGROUND", (2,0), (2,-1), colors.HexColor("#dbeafe")),
        ("FONTNAME",   (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("TEXTCOLOR",  (0,0), (0,-1),  colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR",  (2,0), (2,-1),  colors.HexColor("#1e3a5f")),
        ("FONTNAME",   (0,0), (0,-1),  "Helvetica-Bold"),
        ("FONTNAME",   (2,0), (2,-1),  "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.HexColor("#f8fafc"), colors.white]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # Diagnosis result
    story.append(Paragraph("Diagnosis Result", h1s))
    result_data = [
        ["Predicted Stage", "Confidence", "Severity Zone", "Severity Score"],
        [desc.get("headline", cls), f"{req.confidence:.1f}%",
         sev_zone.get("zone", ""), f"{req.severity} / 100"],
    ]
    rt = Table(result_data, colWidths=[5*cm, 3.5*cm, 4*cm, 3.5*cm])
    rt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("BACKGROUND", (0,1), (-1,1), colors.HexColor(clr_hex + "33")),
        ("TEXTCOLOR",  (0,1), (-1,1), clr),
        ("FONTNAME",   (0,1), (-1,1), "Helvetica-Bold"),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f0f9ff")]),
    ]))
    story.append(rt)
    story.append(Spacer(1, 8))

    # Clinical description
    story.append(Paragraph("Clinical Description", h1s))
    story.append(Paragraph(desc.get("definition", ""), body))
    story.append(Paragraph(desc.get("severity_explanation", ""), body))
    story.append(Spacer(1, 6))

    # Symptoms
    story.append(Paragraph("Reported / Expected Symptoms", h1s))
    for s in desc.get("symptoms", []):
        story.append(Paragraph(f"• {s}", body))
    story.append(Spacer(1, 6))

    # Precautions
    story.append(Paragraph("Precautions & Recommendations", h1s))
    for p in desc.get("precautions", []):
        story.append(Paragraph(f"• {p}", body))
    story.append(Spacer(1, 6))

    # Diet
    story.append(Paragraph("Diet & Health Advice", h1s))
    for d in desc.get("diet", []):
        story.append(Paragraph(f"• {d}", body))
    story.append(Spacer(1, 6))

    # Survey results
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1")))
    story.append(Paragraph("Survey Results", h1s))
    story.append(Paragraph(
        f"Survey Score: {req.survey_score} / 20 &nbsp; | &nbsp; Evaluation: {req.survey_evaluation}", body))
    story.append(Spacer(1, 4))

    # Survey answers table
    opt_labels = ["No", "Sometimes", "Yes"]
    survey_rows = [["#", "Question", "Answer"]]
    for i, q in enumerate(questions):
        if i < len(req.survey_answers):
            survey_rows.append([
                str(i+1),
                q["text"],
                opt_labels[min(req.survey_answers[i], 2)]
            ])
    st = Table(survey_rows, colWidths=[1*cm, 12*cm, 3*cm])
    st.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#e2e8f0")),
        ("PADDING", (0,0), (-1,-1), 5),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(st)
    story.append(Spacer(1, 8))

    # Doctor recommendations
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1")))
    story.append(Paragraph("Recommended Medical Action", h1s))
    story.append(Paragraph(f"Urgency: {doctors.get('urgency','')}", warn if "URGENT" in doctors.get("urgency","") else body))
    story.append(Paragraph(doctors.get("message",""), body))
    story.append(Spacer(1, 4))

    spec = ", ".join(doctors.get("specialist_types", []))
    story.append(Paragraph(f"Recommended Specialists: {spec}", body))
    story.append(Spacer(1, 4))

    for clinic in doctors.get("clinics", []):
        story.append(Paragraph(
            f"• <b>{clinic['name']}</b> — {clinic['dept']}, {clinic['city']} | {clinic['url']}", body))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Online Platforms (Lahore):", body))
    for p in doctors.get("platforms", []):
        story.append(Paragraph(f"• <b>{p['name']}</b>: {p['desc']} — {p['url']}", body))

    # Final recommendation
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1e3a5f")))
    story.append(Paragraph("Final Recommendation", h1s))
    story.append(Paragraph(desc.get("recommendation", "Please consult a medical professional."), body))

    # Disclaimer
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "⚠ DISCLAIMER: This report is generated by an AI research tool and is NOT a medical diagnosis. "
        "Always consult a qualified neurologist or physician for clinical assessment and treatment decisions.",
        small))

    doc.build(story)
    return buf.getvalue()

# ════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "running", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accept MRI image → return prediction, Grad-CAM, descriptions, survey Qs, doctors."""
    if model is None:
        raise HTTPException(503, "Model not loaded. Place maxvit_best_model.pth in backend/")
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not open image.")

    # Also encode original image as base64 for display in frontend
    orig_buf = io.BytesIO()
    image.save(orig_buf, format="PNG")
    orig_b64 = "data:image/png;base64," + base64.b64encode(orig_buf.getvalue()).decode()

    # Preprocess + infer
    tensor  = transform(image).unsqueeze(0).to(device)
    output  = model(tensor)
    probs   = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
    pred_idx   = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]

    # Grad-CAM
    gradcam_b64 = get_gradcam(tensor, pred_idx)

    desc    = DESCRIPTIONS[pred_class]
    szone   = SEVERITY_ZONE[pred_class]
    doctors = DOCTORS[pred_class]
    questions = SURVEY_QUESTIONS[pred_class]

    return {
        # Core prediction
        "predicted":    pred_class,
        "label":        szone["label"],
        "color":        szone["color"],
        "confidence":   round(float(probs[pred_idx]) * 100, 1),
        "severity":     SEVERITY[pred_class],
        "severity_zone": szone["zone"],

        # Images
        "original_mri": orig_b64,
        "gradcam":      gradcam_b64,

        # Class probabilities (all 4)
        "all": [
            {"name": CLASS_NAMES[i],
             "label": SEVERITY_ZONE[CLASS_NAMES[i]]["label"],
             "color": SEVERITY_ZONE[CLASS_NAMES[i]]["color"],
             "percentage": round(float(probs[i]) * 100, 1)}
            for i in range(len(CLASS_NAMES))
        ],

        # Description
        "description": desc,

        # Survey questions (class-specific)
        "survey_questions": questions,

        # Doctor recommendations
        "doctors": doctors,
    }


@app.post("/survey")
def survey(req: SurveyRequest):
    """Calculate survey score from answers and return risk evaluation."""
    questions = SURVEY_QUESTIONS.get(req.predicted_class, [])
    if not questions:
        raise HTTPException(400, f"Unknown class: {req.predicted_class}")

    score = 0
    for i, q in enumerate(questions):
        if i >= len(req.answers):
            break
        ans = min(max(req.answers[i], 0), 2)   # clamp 0–2
        if q["type"] == "risk":
            # Yes=2 (bad), Sometimes=1, No=0
            score += ans
        else:
            # protective: Yes=0 (good), Sometimes=1, No=2 (bad)
            score += (2 - ans)

    max_score = len(questions) * 2   # 10 × 2 = 20

    if score <= 6:
        evaluation = "Stable"
        eval_color = "#22c55e"
        eval_msg   = "Your survey responses indicate a stable condition. Keep maintaining healthy habits and continue regular check-ups."
    elif score <= 12:
        evaluation = "At Risk"
        eval_color = "#f97316"
        eval_msg   = "Your responses suggest some risk factors are present. Consultation with a specialist and lifestyle changes are recommended."
    else:
        evaluation = "Needs Attention"
        eval_color = "#ef4444"
        eval_msg   = "Multiple concerning factors identified. Immediate medical consultation and a care plan are strongly recommended."

    return {
        "score":       score,
        "max_score":   max_score,
        "percentage":  round(score / max_score * 100),
        "evaluation":  evaluation,
        "eval_color":  eval_color,
        "eval_msg":    eval_msg,
    }


@app.post("/generate-pdf")
def generate_pdf(req: PDFRequest):
    """Generate and return a downloadable PDF report."""
    try:
        pdf_bytes = build_pdf(req)
    except Exception as e:
        raise HTTPException(500, f"PDF generation failed: {str(e)}")

    fname = f"NeuroScan_Report_{req.predicted_class}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={fname}"}
    )
