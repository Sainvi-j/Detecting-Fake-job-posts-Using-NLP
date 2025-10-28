# task3 correlation insights
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# 1. Load cleaned data
df = pd.read_csv('preprocessed_fake_job_postings.csv')          # contains clean_description
df = df[['has_company_logo','telecommuting','employment_type',
         'required_experience','fraudulent','clean_description']].copy()

# 2. Value counts grouped by fraudulent
cols = ['has_company_logo','telecommuting','employment_type','required_experience']

print("\n=== Value counts grouped by fraudulent ===")
for c in cols:
    print(f"\n--- {c} ---")
    print(df.groupby('fraudulent')[c].value_counts(normalize=True).round(3))

# 3. Bar charts (proportion of fake vs real)
sns.set_style('whitegrid')
fig, axs = plt.subplots(1,3, figsize=(15,5))

# Logo 
logo_prop = df.groupby('fraudulent')['has_company_logo'].value_counts(normalize=True).unstack()
logo_prop.plot(kind='bar', ax=axs[0], color=['#66c2a5','#fc8d62'])
axs[0].set_title('Company Logo')
axs[0].set_ylabel('Proportion')
axs[0].legend(title='has_logo', labels=['No','Yes'])

# Remote
remote_prop = df.groupby('fraudulent')['telecommuting'].value_counts(normalize=True).unstack()
remote_prop.plot(kind='bar', ax=axs[1], color=['#8da0cb','#e78ac3'])
axs[1].set_title('Telecommuting')
axs[1].legend(title='remote', labels=['No','Yes'])

# Employment type (top 5 only)
top_emp = df['employment_type'].value_counts().head(5).index
emp_prop = (df[df['employment_type'].isin(top_emp)]
            .groupby('fraudulent')['employment_type']
            .value_counts(normalize=True).unstack())
emp_prop.plot(kind='bar', ax=axs[2], stacked=False)
axs[2].set_title('Employment Type (top 5)')
axs[2].legend(title='type')

plt.tight_layout()
plt.show()

# 4. WordClouds – real vs fake
real_text = ' '.join(df[df['fraudulent']==0]['clean_description'].fillna(''))
fake_text = ' '.join(df[df['fraudulent']==1]['clean_description'].fillna(''))

wc_real = WordCloud(width=800, height=400, background_color='white',
                    colormap='Greens', max_words=150).generate(real_text)
wc_fake = WordCloud(width=800, height=400, background_color='white',
                    colormap='Reds',   max_words=150).generate(fake_text)

fig, axs = plt.subplots(1,2, figsize=(16,6))
axs[0].imshow(wc_real, interpolation='bilinear')
axs[0].set_title('Real Job Descriptions', fontsize=16)
axs[0].axis('off')

axs[1].imshow(wc_fake, interpolation='bilinear')
axs[1].set_title('Fake Job Descriptions', fontsize=16)
axs[1].axis('off')

plt.show()