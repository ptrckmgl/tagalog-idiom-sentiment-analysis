for row in soup.select("div.views-row:has(div.views-field-node-normalized-title a)"):
    title = row.find("a").text.strip()
    pos_el = row.find(class_="part-speech-field-content")
    
    # 1. Find all <li> tags inside the row
    li_tags = row.find_all("li")
    
    pos = pos_el.text.strip() if pos_el else "N/A"
    
    # 2. Join all li text items with \n (or fallback if no <li> tags exist)
    if li_tags:
        mean = "\n".join(li.text.strip() for li in li_tags)
    else:
        mean_el = row.find(class_="views-field-field-definition-translation")
        mean = mean_el.text.strip() if mean_el else "N/A"
    
    print(f"\"{title}\"\n {pos}\n {mean}\n--------------------------------------------")