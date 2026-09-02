import re
from datetime import datetime

# Palavras-chave que indicam urgência
URGENT_KEYWORDS = ["urgente", "imediato", "prazo", "vencimento", "bloqueio", "hoje", "intimação"]

# Padrões comuns de datas no formato brasileiro (DD/MM/AAAA, DD/MM/AA, DD-MM-AAAA)
DATE_PATTERN = re.compile(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b')

def parse_date_br(date_str: str) -> datetime | None:
    for fmt in ('%d/%m/%Y', '%d/%m/%y', '%d-%m-%Y', '%d-%m-%y'):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            pass
    return None

def calculate_priority_score(texto: str, reference_date: datetime | None = None) -> float:
    """
    Calcula um score de prioridade de 0.0 a 1.0.
    - +0.2 para cada palavra-chave de urgência encontrada (max 0.6)
    - +0.5 se houver data atrasada, para hoje ou amanhã
    - +0.3 se houver data para os próximos 3 dias
    - +0.1 se houver data para os próximos 7 dias
    """
    score = 0.0
    texto_lower = texto.lower()
    
    # 1. Checagem de palavras-chave
    keyword_count = sum(1 for word in URGENT_KEYWORDS if word in texto_lower)
    score += min(keyword_count * 0.2, 0.6)
    
    # 2. Checagem de datas
    ref_date = reference_date or datetime.now()
    dates_found = []
    
    for match in DATE_PATTERN.finditer(texto):
        date_str = match.group(0)
        parsed = parse_date_br(date_str)
        if parsed:
            # Assumir que datas com ano muito fora não são reais ou são antigas demais (ex: 2000)
            if parsed.year < 2000 or parsed.year > 2100:
                continue
            dates_found.append(parsed)
            
    if dates_found:
        max_date_score = 0.0
        for d in dates_found:
            diff_days = (d.date() - ref_date.date()).days
            
            if diff_days <= 1:
                date_score = 0.5
            elif diff_days <= 3:
                date_score = 0.3
            elif diff_days <= 7:
                date_score = 0.1
            else:
                date_score = 0.0
                
            max_date_score = max(max_date_score, date_score)
            
        score += max_date_score
        
    return min(1.0, score)

def get_priority_label(score: float) -> str:
    if score >= 0.7:
        return "Alta"
    elif score >= 0.4:
        return "Média"
    else:
        return "Baixa"
