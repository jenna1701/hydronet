import requests
import pathlib
# pip install fair-research-login
import fair_research_login

# Use the following in the globus-cli to find the base collection HTTPS URL
# globus collection show f58973c0-08c1-43a7-9a0e-71f54ddc973c

myfile = 'https://g-83fdd0.1beed.03c0.data.globus.org/static_datasets/W3-W30_all_geoms_TTM2.1-F.zip'

# ID for PNNL DataHub collection
collection = 'f58973c0-08c1-43a7-9a0e-71f54ddc973c'

# Guest Collections only require the https scope.
scopes = [f'https://auth.globus.org/scopes/{collection}/https']

# Fetch an HTTPS token
client = fair_research_login.NativeClient(client_id='7414f0b4-7d05-4bb6-bb00-076fa3f17cf5')
tokens = client.login(requested_scopes=scopes)
# User is given a URL to visit, leads to Globus log via institution, Google, or ORCID, returns a token

https_token = tokens[collection]['access_token']

# Fetch the file
filename = pathlib.Path(myfile).name
headers = {'Authorization': f'Bearer {https_token}', 'Content-Disposition': 'download'}
with requests.get(myfile, stream=True, headers=headers) as r:
    r.raise_for_status()
    with open(filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=4096): 
            f.write(chunk)

print(f'Downloaded {myfile}')

