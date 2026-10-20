from bs4 import BeautifulSoup
import csv
import requests

url = "https://www.leksyon.com/filipino-idioms/y"
response = requests.get(url)

soup = BeautifulSoup(response.content, "html.parser")   

with open("y_idioms.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Title", "Part of Speech", "Meaning"])
    
    for row in soup.select("div.views-row:has(div.views-field-node-normalized-title a)"):
        title = row.find("a").text.strip()
        pos_el = row.find(class_="part-speech-field-content")
        
        li_tags = row.find_all("li")
        pos = pos_el.text.strip() if pos_el else "N/A"
        
        if li_tags:
            mean = "\n".join(li.text.strip() for li in li_tags)
        else:
            mean_el = row.find(class_="views-field-field-definition-translation")
            mean = mean_el.text.strip() if mean_el else "N/A"
        
        writer.writerow([title, pos, mean])

