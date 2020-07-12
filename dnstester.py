import glob
import os
import time
from datetime import datetime

import click
import dns.resolver
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def sep_colom(df, sep_colom):
    sep = []
    for rep in np.unique(df[sep_colom]):
        mask = df[sep_colom] == rep
        sep.append(df[mask])
    return sep


resolver = dns.resolver.Resolver()

reps = click.prompt("Enter the number of repetitions", type=int, default=20)
rep_sleep = click.prompt("Enter the amount of sleep between repetitions (sec)", type=float, default=1.0)
urls_per_rep = click.prompt("Enter the number of urls for each repetition", type=int, default=100)

urls_file = "config/websites/domains.csv"
dns_file = "config/dns/dns.csv"
out_folder = "dnsout"

change_config = click.prompt("Change extended configs", type=bool, default=False)
if change_config:
    urls_file = click.prompt("Enter websites cvs file path", type=str, default=urls_file)
    dns_file = click.prompt("Enter dns cvs file path", type=str, default=dns_file)
    out_folder = click.prompt("Enter out folder", type=str, default=out_folder)

urls = np.concatenate(pd.read_csv(urls_file).to_numpy())
dns_servers_df = pd.read_csv(dns_file)

os.makedirs(out_folder, exist_ok=True)
plot_folder = os.path.join(out_folder, "imgs")

if os.path.exists(plot_folder):
    for f in glob.glob(os.path.join(plot_folder, "*.png")):
        os.remove(f)
else:
    os.makedirs(plot_folder, exist_ok=True)

dns_servers = []
for i in range(len(dns_servers_df)):
    dns_info = dns_servers_df.iloc[i]
    servers = dns_info["servers"].split(",")
    dns_name = dns_info["name"]
    dns_servers.append([dns_name, servers])

dns_servers = np.array(dns_servers, dtype=object)

result_df = pd.DataFrame(columns=['rep', 'time_stamp', 'dns', 'answer_time', 'url'])

for i in tqdm(range(reps), ascii=True):
    rep_urls = np.random.choice(urls, size=urls_per_rep, replace=True)

    np.random.shuffle(dns_servers)

    for dns_name, servers in dns_servers:
        resolver.nameservers = servers

        for url in rep_urls:
            try:
                answer = resolver.query(url)
            except:
                print("No answer for:", url, "via DNS:", dns_name)
            else:
                answer_time_ms = answer.response.time * 1000

                result_df = result_df.append(
                    {'rep': i, 'time_stamp': str(datetime.now()), 'dns': dns_name, "answer_time": answer_time_ms,
                     'url': url},
                    ignore_index=True)

    result_df.to_csv(os.path.join(out_folder, "result.csv"), index=False)
    time.sleep(rep_sleep)

plot_data = []
for dns_name in np.unique(result_df["dns"]):
    dns_data = result_df[result_df["dns"] == dns_name]

    answer_time_resp = np.array([el["answer_time"].to_numpy() for el in sep_colom(dns_data, "rep")], dtype=object)
    answer_time_resp_mean = np.mean(answer_time_resp, axis=1)
    title = f"{dns_name}: {np.mean(answer_time_resp):0.2f} ms"

    plot_data.append([title, dns_name, answer_time_resp])  # answer_time_resp has to be last or chance -1 later

plot_data = np.array(plot_data, dtype=object)
sort_idx = np.argsort([np.mean(el) for el in plot_data[:, -1]])
plot_data = plot_data[sort_idx]

for i, (title, dns_name, answer_time_resp) in enumerate(plot_data):
    y = answer_time_resp.mean(axis=1)
    x = np.arange(len(y))

    plt.figure()
    plt.title(title)
    plt.plot(x, y, "-o")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f"{i}_{dns_name}.png"), dpi=300)
    plt.close()
