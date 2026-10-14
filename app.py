"""CRM dashboard for deployment with Gunicorn."""
from pathlib import Path
import json
import pickle
from html import escape
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, ctx, no_update, dcc, html


data_dir = Path(__file__).resolve().parent
file_path = data_dir / 'data.pickle'
with open(file_path, 'rb') as data_file:
    calls, contacts, deals, spend = pickle.load(data_file)
deals, calls = deals.copy(), calls.copy()
deals = deals.loc[~deals['Product'].astype('string').str.strip().str.casefold()
    .eq('find yourself in it').fillna(False)].copy()
deals['paid'] = deals['Stage'].eq('Payment Done') | deals['Initial Amount Paid'].gt(0)
deals['Результат'] = np.where(deals['paid'], 'Оплачено', np.where(deals['Stage'].eq('Lost'), 'Отказы', 'В работе'))
filter_fields = ['Product', 'Source', 'Deal Owner Name', 'Результат', 'German Level']
for field in filter_fields:
    deals[field] = deals[field].astype('string').str.strip().replace('', pd.NA).fillna('Не указано')
deals['_date'] = pd.to_datetime(deals['Created Time'], format='%d.%m.%Y %H:%M').dt.normalize()
deals['_contact'] = deals['Contact Name'].astype('string').str.strip().replace('', pd.NA)
calls['_date'] = pd.to_datetime(calls['Call Start Time'], format='%d.%m.%Y %H:%M').dt.normalize()
calls['_contact'] = calls['CONTACTID'].astype('string').str.strip().replace('', pd.NA)
period_start = max(deals['_date'].min(), calls['_date'].min())
period_end = min(deals['_date'].max(), calls['_date'].max())
calendar = pd.date_range(period_start, period_end, freq='D')
months = sorted(calendar.strftime('%Y-%m').unique().tolist())
ALL = '__all__'
COLORS = {'Сделки': '#7F70D5', 'Звонки': '#55BFA8'}
options = {field: [{'label': 'Все', 'value': ALL}] + [
    {'label': value, 'value': value}
    for value in sorted(deals[field].unique(), key=lambda v: (v == 'Не указано', v))]
    for field in filter_fields}

# Use the user's existing coordinate file; no geocoding or external lookup.
coordinates_path = data_dir / 'city_data_google_en.json'
city_records = json.loads(coordinates_path.read_text(encoding='utf-8-sig'))
coordinates, city_aliases = {}, {}
for name, record in city_records.items():
    if not isinstance(record, dict):
        continue
    lat, lon = record.get('latitude'), record.get('longitude')
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        continue
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        continue
    coordinates[str(name).strip().casefold()] = (lat, lon)
    if record.get('city'):
        city_aliases.setdefault(str(record['city']).strip().casefold(), set()).add((lat, lon))
for alias, positions in city_aliases.items():
    if len(positions) == 1:
        coordinates.setdefault(alias, next(iter(positions)))
city_keys = deals['City'].astype('string').str.strip().str.casefold()
deals['_lat'] = city_keys.map({city: position[0] for city, position in coordinates.items()})
deals['_lon'] = city_keys.map({city: position[1] for city, position in coordinates.items()})

closed_data = None
call_workload = None


def closed_animation(selected, mode):
    global closed_data, call_workload
    if closed_data is None:
        closed_data = deals.copy()
        closed_data['Stage'] = closed_data['Stage'].mask(closed_data['paid'], 'Payment Done')
        closed_data['_created_at'] = pd.to_datetime(closed_data['Created Time'], format='%d.%m.%Y %H:%M')
        closed_data['_closed_at'] = pd.to_datetime(closed_data['Closing Date'], format='%d.%m.%Y')
        closed_data['Days to Close'] = (closed_data['_closed_at'].dt.normalize()
            - closed_data['_created_at'].dt.normalize()).dt.days
        closed_data = closed_data.loc[closed_data['Stage'].isin(['Lost', 'Payment Done'])
            & closed_data['Days to Close'].ge(0)].copy()
    if mode in ['calls', 'minutes'] and call_workload is None:
        # Same interval calculation as section 9.2.4 of notebook 2.
        work = closed_data.loc[closed_data['_contact'].notna()].copy()
        work['Call Count'], work['Call Minutes'] = 0, 0.0
        groups = work.groupby('_contact').groups
        call_rows = calls.dropna(subset=['_contact', 'Call Start Time']).copy()
        call_rows['_at'] = pd.to_datetime(call_rows['Call Start Time'], format='%d.%m.%Y %H:%M')
        call_rows['_seconds'] = pd.to_numeric(call_rows['Call Duration (in seconds)'])
        for contact, group in call_rows.groupby('_contact', sort=False):
            if contact not in groups:
                continue
            indices = groups[contact]
            group = group.sort_values('_at')
            times = group['_at'].to_numpy()
            cumulative = np.r_[0, group['_seconds'].fillna(0).cumsum()]
            left = times.searchsorted(work.loc[indices, '_created_at'].to_numpy())
            right = times.searchsorted((work.loc[indices, '_closed_at'].dt.normalize()
                + pd.Timedelta(days=1)).to_numpy())
            work.loc[indices, 'Call Count'] = right - left
            work.loc[indices, 'Call Minutes'] = (cumulative[right] - cumulative[left]) / 60
        call_workload = work[['Call Count', 'Call Minutes']]
    data = closed_data.loc[closed_data.index.isin(selected.index)].copy()
    if mode in ['calls', 'minutes']:
        data = data.loc[data['_contact'].notna()].join(call_workload)
    data = data.sort_values('_closed_at')
    size_column = {'duration': 'Days to Close', 'calls': 'Call Count', 'minutes': 'Call Minutes'}[mode]
    animation_months = sorted(data['_closed_at'].dt.strftime('%Y-%m').unique().tolist())
    if data.empty:
        return {'data': [], 'layout': {'height': 650, 'paper_bgcolor': '#ffffff', 'plot_bgcolor': '#ffffff'}}
    max_days = float(data['Days to Close'].max())
    max_size = float(data[size_column].max())
    size_ref = 2 * max(1, max_size) / (55 if mode == 'duration' else 45)**2
    ticks = np.unique(np.geomspace(1, max_days + 1, 8).round().astype(int)) - 1
    animation_frames = []
    for month in animation_months:
        traces = []
        for stage, label, color in [('Payment Done', 'Оплачено', '#2CA25F'), ('Lost', 'Отказы', '#E15759')]:
            part = data.loc[data['_closed_at'].dt.strftime('%Y-%m').eq(month) & data['Stage'].eq(stage)]
            details = [part['Id'].astype(str).tolist(), part['_created_at'].dt.strftime('%d.%m.%Y').tolist(),
                part['_closed_at'].dt.strftime('%d.%m.%Y').tolist(), part['Product'].tolist(),
                part['Days to Close'].tolist()]
            hover = ('<b>%{customdata[0]}</b><br>' + label
                + '<br>Создание: %{customdata[1]}<br>Закрытие: %{customdata[2]}'
                + '<br>Продукт: %{customdata[3]}<br>Срок: %{customdata[4]:.0f} дней')
            if mode in ['calls', 'minutes']:
                details += [part['Call Count'].tolist(), part['Call Minutes'].tolist()]
                hover += '<br>Звонков: %{customdata[5]:.0f}<br>Минут: %{customdata[6]:.1f}'
            traces.append({'type': 'scatter', 'mode': 'markers', 'name': label, 'legendgroup': label,
                'x': part['_closed_at'].dt.day.tolist(), 'y': np.log1p(part['Days to Close'] / 7).tolist(),
                'ids': part['Id'].astype(str).tolist(), 'customdata': list(map(list, zip(*details))),
                'marker': {'size': part[size_column].tolist(), 'sizemode': 'area', 'sizeref': size_ref,
                    'sizemin': 4, 'color': color, 'opacity': .75, 'line': {'color': '#ffffff', 'width': .5}},
                'hovertemplate': hover + '<extra></extra>'})
        animation_frames.append({'name': month, 'data': traces, 'traces': [0, 1]})
    instant = {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': True}, 'transition': {'duration': 0}}
    return {'data': animation_frames[0]['data'], 'frames': animation_frames, 'layout': {
        'height': 650, 'autosize': True, 'paper_bgcolor': '#ffffff', 'plot_bgcolor': '#ffffff',
        'font': {'family': 'Arial, sans-serif', 'size': 12, 'color': '#74747f'},
        'margin': {'l': 65, 'r': 25, 't': 45, 'b': 125},
        'xaxis': {'title': {'text': 'День закрытия'}, 'range': [0, 32], 'dtick': 1, 'gridcolor': '#f2f1f6'},
        'yaxis': {'title': {'text': 'Дни до закрытия'}, 'range': [-.02, max(.1, float(np.log1p(max_days/7))*1.1)],
            'tickvals': np.log1p(ticks / 7).tolist(), 'ticktext': ['<1 дня' if t == 0 else str(t) for t in ticks],
            'gridcolor': '#f2f1f6'},
        'legend': {'orientation': 'h', 'x': 0, 'y': 1.1},
        'sliders': [{'active': 0, 'x': .12, 'len': .88, 'y': 0, 'pad': {'t': 55},
            'currentvalue': {'visible': False}, 'steps': [{'label': m, 'method': 'animate',
                'args': [[m], instant]} for m in animation_months]}],
        'updatemenus': [{'type': 'buttons', 'direction': 'left', 'showactive': False,
            'x': 0, 'y': 0, 'xanchor': 'left', 'yanchor': 'top', 'pad': {'t': 55},
            'buttons': [{'label': '▶', 'method': 'animate', 'args': [None, {**instant,
                'fromcurrent': True, 'frame': {'duration': 700, 'redraw': True}}]},
                {'label': 'Ⅱ', 'method': 'animate', 'args': [[None], instant]}]}],
    }}

app = Dash(__name__, title='Анализ данных CRM онлайн-школы программирования X')
server = app.server
app.index_string = '''<!DOCTYPE html>
<html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
*{box-sizing:border-box}body{margin:0;background:#fff;color:#303432;font-family:Arial,sans-serif}
.dashboard{max-width:1200px;margin:auto;padding:26px 24px 18px}
h1{font-size:26px;font-weight:600;margin:0 0 22px;letter-spacing:-.5px}
.workspace{display:grid;grid-template-columns:120px minmax(0,1fr);gap:26px}
.map-wheel-row{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:20px}.map-wheel-row>.chart-panel{min-width:0;padding:8px 6px 0}
.filters{padding:4px 18px 0 0;border-right:1px solid #ededf1}
.filter{margin-bottom:22px}.filter>label{display:block;font-size:12px;color:#777b82;margin-bottom:8px}
.reset{background:white;border:1px solid #d9d9e2;border-radius:7px;color:#646079;padding:9px 12px;width:100%;cursor:pointer}
.products{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px}
.products>label{padding:9px 15px;border:1px solid #d8d7df;border-radius:7px;font-size:13px;cursor:pointer;background:#fff}
.products>label:has(input:checked){background:#8373d7;border-color:#8373d7;color:#fff}
.products input{position:absolute;opacity:0;width:0;height:0}
.metrics{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));border-top:1px solid #eeedf2;padding:18px 0 22px;gap:16px}
.metric-label{font-size:13px;color:#7c7f86;margin-bottom:6px}.metric-value{font-size:32px;font-weight:600;letter-spacing:-.8px;font-variant-numeric:tabular-nums}
#remaining-value{font-size:24px}
.chart-panel{border:1px solid #eeedf2;border-radius:12px;padding:16px 14px 4px;box-shadow:0 1px 3px #29263c05}
.animation-tabs{margin-bottom:12px}.animation-tab{padding:12px 8px!important;font-size:13px!important;border:0!important;border-bottom:2px solid #eeedf2!important;background:white!important}
.animation-tab-selected{color:#7360c8!important;border-bottom:2px solid #8373d7!important}
h2{font-size:14px;font-weight:400;margin:0 0 5px;color:#6b6b75}
.dash-dropdown{font-size:12px}.Select-control{border-radius:7px!important;border-color:#e1dfe8!important}
@media(max-width:760px){.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.map-wheel-row{grid-template-columns:1fr}.dashboard{padding:18px 12px}.workspace{grid-template-columns:1fr;gap:16px}.filters{display:grid;grid-template-columns:1fr 1fr;gap:12px;border:0;padding:0}.filter{margin:0}.reset{align-self:end}.metric-value{font-size:27px}h1{font-size:23px}}
</style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>'''

filter_controls = []
for field, label, control_id in [
    ('Source', 'Источник', 'source'),
    ('Deal Owner Name', 'Менеджер', 'owner'),
    ('Результат', 'Результат', 'result'),
    ('German Level', 'Немецкий язык', 'german'),
]:
    filter_controls.append(html.Div([
        html.Label(label, htmlFor=control_id),
        dcc.Dropdown(id=control_id, options=options[field], value=ALL, clearable=False,
                     searchable=True, placeholder='Все', className='filter-dropdown'),
    ], className='filter'))
filter_controls.append(html.Button('Сбросить фильтры', id='reset', n_clicks=0, className='reset'))

cards = []
for title, value_id in [('Контрактов', 'deals-value'), ('Оплачено', 'paid-value'), ('Конверсия в оплату', 'conversion-value'),
                        ('Студентов', 'students-value'), ('Ещё получим', 'remaining-value')]:
    cards.append(html.Div([
        html.Div(title, className='metric-label'),
        html.Div(id=value_id, className='metric-value'),
    ]))
product_options = [{'label': 'Все продукты', 'value': ALL}] + [option for option in options['Product'][1:] if option['value'] != 'Не указано']
def journey_chart(selected):
    full_deals = globals()["deals"]
    created = full_deals['_date'].min()
    closed = pd.to_datetime(full_deals.loc[full_deals['paid'] | full_deals['Stage'].eq('Lost'), 'Closing Date'], format='%d.%m.%Y').max()
    AVERAGE_PERIODS = {'В среднем в месяц': max(1, (closed.normalize() - created).days + 1) / 30}
    deals = selected.copy().replace({'Не указано': pd.NA})
    if deals.empty:
        return {'data': [], 'layout': {'height': 650}}
    sales = deals.copy()
    sales['sale_amount'] = sales['Initial Amount Paid'].where(sales['paid'])
    sales['first_payment'] = sales['Initial Amount Paid'].where(sales['paid'])
    # Сводный рейтинг рекламных факторов: число сделок и нижняя граница конверсии по Уилсону.
    journey_factors = ["Source", "Campaign", "Term", "Content"]
    factor_values = deals[journey_factors].copy()
    factor_values["_paid"] = sales["paid"]
    journey_rank = factor_values.groupby(journey_factors, dropna=False).agg(
        Сделок=("_paid", "size"), Оплачено=("_paid", "sum")
    ).reset_index()
    journey_rank = journey_rank.loc[
        journey_rank["Сделок"].ge(20)
        & journey_rank[journey_factors].notna().sum(axis=1).ge(2)
    ].copy()
    n, p = journey_rank["Сделок"], journey_rank["Оплачено"] / journey_rank["Сделок"]
    z = 1.959963984540054
    journey_rank["CR_low"] = (
        (p + z**2 / (2*n) - z*np.sqrt(p*(1-p)/n + z**2/(4*n**2)))
        / (1 + z**2/n)
    ).clip(0, 1) * 100
    journey_rank["Средний балл"] = (
        journey_rank["Сделок"].rank(ascending=False)
        + journey_rank["CR_low"].rank(ascending=False)
    ) / 2
    journey_rank = journey_rank.sort_values(
        ["Средний балл", "Оплачено", "Сделок"],
        ascending=[True, False, False], kind="stable"
    ).reset_index(drop=True)
    journey_rank["Место"] = np.arange(1, len(journey_rank) + 1)
    def factor_key(values):
        return tuple(None if pd.isna(v) else str(v) for v in values)

    combination_labels, combination_descriptions, combination_places = {}, {}, {}
    for _, row in journey_rank.head(10).iterrows():
        key = factor_key(row[journey_factors])
        rank = int(row["Место"])
        desc = "<br>".join(f"{field}: {escape(value)}"
                          for field, value in zip(journey_factors, key) if value is not None)
        label = f"#{rank} · " + " / ".join(
            value if len(value) <= 24 else value[:21] + "..."
            for value in key if value is not None
        )
        combination_labels[key] = label
        combination_descriptions[label] = f"<b>Место в «Сводном рейтинге рекламных факторов»: {rank}</b><br>{desc}<br><br>"
        combination_places[label] = rank

    journey = deals[["Page", "Deal Owner Name", "Stage", "Product",
                     "Education Type", "Lost Reason"]].copy()
    journey["Результат"] = np.where(
        sales["paid"], "Оплачено",
        np.where((journey["Stage"] == "Lost").fillna(False), "Отказы", "В работе")
    )
    product = journey["Product"].copy()
    schedule = journey["Education Type"]
    with_schedule = product.notna() & schedule.isin(["Morning", "Evening"])
    product.loc[with_schedule] += " · " + schedule.loc[with_schedule]
    journey["Далее"] = product.where((journey["Результат"] == "Оплачено"))
    for outcome, field in [("Отказы", "Lost Reason"), ("В работе", "Stage")]:
        mask = (journey["Результат"] == outcome)
        journey.loc[mask, "Далее"] = journey.loc[mask, field]
    for field in ["Page", "Deal Owner Name"]:
        leaders = journey[field].value_counts().head(10).index
        mask = journey[field].notna() & ~journey[field].isin(leaders)
        journey.loc[mask, field] = "Другие"

    journey["Сочетание"] = [
        combination_labels.get(factor_key(values), "Другие")
        if any(pd.notna(v) for v in values) else pd.NA
        for values in factor_values[journey_factors].itertuples(index=False, name=None)
    ]

    # Узлы различаются по колонке и ветке результата.
    node_totals, edge_totals = {}, {}
    def add_flow(target, key, values):
        target[key] = target.get(key, np.zeros(4)) + values

    for idx, row in journey.iterrows():
        outcome = row["Результат"]
        path = [(0, "", row["Сочетание"]), (1, "", row["Page"]),
                (2, "", row["Deal Owner Name"]), (3, outcome, row["Далее"]),
                (4, "", outcome)]
        path = [key for key in path if pd.notna(key[2])]
        values = np.array([1, int(sales.at[idx, "paid"]),
                           sales.at[idx, "sale_amount"],
                           sales.at[idx, "first_payment"]], dtype=float)
        # В суммах участвуют только известные значения.
        values[2:] = np.nan_to_num(values[2:])
        for key in path:
            add_flow(node_totals, key, values)
        for source, target in zip(path, path[1:]):
            add_flow(edge_totals, (source, target), values)

    keys = sorted(node_totals, key=lambda k: (
        k[0], 1 if k[2] == "Другие" else 0,
        {"Оплачено": 0, "В работе": 1, "Отказы": 2}.get(k[1] or k[2], 3),
        combination_places.get(k[2], 999) if k[0] == 0 else -node_totals[k][0], k[2]
    ))
    node_id = {key: i for i, key in enumerate(keys)}

    # Фиксируем порядок: «Другие» находятся внизу каждой колонки.
    node_y = {}
    gap = 18 / (1700 - 126 - 30)
    columns = [[k for k in keys if k[0] == col] for col in range(5)]
    height_scale = min(
        (1 - gap * (len(group) - 1)) / sum(node_totals[k][0] for k in group)
        for group in columns if group
    )
    for group in columns:
        cursor = 0.0
        for key in group:
            height = node_totals[key][0] * height_scale
            node_y[key] = cursor + height / 2
            cursor += height + gap

    outcome_colors = {"Оплачено": "#15B8A6", "Отказы": "#E66DB1",
                      "В работе": "#FF9F1C"}
    def journey_color(key):
        if key[2] == "Другие":
            return "#B4BAC4"
        return (outcome_colors[key[1]] if key[0] == 3 else
                outcome_colors[key[2]] if key[0] == 4 else
                ["#FF9F1C", "#5B8FF9", "#A855F7"][key[0]])

    def hover_values(values):
        count, paid, amount, first = values
        months = AVERAGE_PERIODS["В среднем в месяц"]
        return (f"Сделок: {count:,.0f}<br>Оплачено: {paid:,.0f}"
                f"<br>Конверсия: {paid / count * 100:.2f}%"
                f"<br>Сумма продаж: {amount:,.0f} €"
                f"<br>Первые платежи: {first:,.0f} €"
                f"<br>В среднем в месяц: {count / months:.2f} сделок;"
                f" {paid / months:.2f} оплат; {amount / months:,.0f} €")

    edges = list(edge_totals)
    journey_fig = go.Figure(go.Sankey(
        arrangement="fixed", valueformat=",.0f", valuesuffix=" сделок",
        node=dict(
            label=[escape(str(k[2])) for k in keys],
            color=[journey_color(k) for k in keys],
            x=[ [0.01, 0.32, 0.55, 0.76, 0.99][k[0]] for k in keys],
            y=[node_y[k] for k in keys],
            pad=18, thickness=16, line=dict(width=0),
            customdata=[combination_descriptions.get(k[2], "") + hover_values(node_totals[k])
                        if k[0] == 0 else hover_values(node_totals[k]) for k in keys],
            hovertemplate="%{label}<br>%{customdata}<extra></extra>"
        ),
        link=dict(
            source=[node_id[a] for a, b in edges],
            target=[node_id[b] for a, b in edges],
            value=[int(edge_totals[e][0]) for e in edges],
            color=["rgba(145,155,175,0.23)"] * len(edges),
            customdata=[hover_values(edge_totals[e]) for e in edges],
            hovertemplate="%{source.label} → %{target.label}<br>"
                          "%{customdata}<extra></extra>"
        )
    ))
    journey_fig.update_layout(
        title=dict(text="Customer Journey: от рекламы до результата",
                   x=0.01, xanchor="left", y=0.99, yanchor="top",
                   yref="container", font=dict(size=18)),
        autosize=True, height=1700, font=dict(size=12, color="#283044"),
        paper_bgcolor="white", margin=dict(l=15, r=15, t=126, b=30),
        annotations=[dict(x=x, y=1, yshift=30, yanchor="bottom",
                          xref="paper", yref="paper",
                          text=f"<b>{label}</b>", showarrow=False, xanchor=anchor,
                          align="left" if anchor == "left" else "center", font=dict(size=12))
                     for x, label, anchor in [
                         (0, "Сводный рейтинг<br>рекламных факторов", "left"),
                         (.32, "Лендинг", "center"), (.55, "Менеджер", "center"),
                         (.76, "Продукт / причина<br>Стадия", "center"),
                         (1, "Результат", "right")]]
    )
    return journey_fig


def workload_chart(selected_deals, owner, restricted):
    all_deals, contacts, calls = globals()['deals'], globals()['contacts'], globals()['calls']
    paid_contacts = all_deals.loc[all_deals['paid'], '_contact'].dropna().unique()
    selected_contacts, selected_calls = contacts, calls
    if restricted:
        ids = selected_deals['_contact'].dropna().unique()
        selected_contacts = contacts.loc[contacts['Id'].astype('string').str.strip().isin(ids)]
        selected_calls = calls.loc[calls['_contact'].isin(ids)]
    if owner != ALL:
        selected_contacts = selected_contacts.loc[selected_contacts['Contact Owner Name'].eq(owner)]
        selected_calls = selected_calls.loc[selected_calls['Call Owner Name'].eq(owner)]
    deals, contacts, calls = selected_deals, selected_contacts, selected_calls
    workload = pd.concat([
        deals.groupby('Deal Owner Name').size().rename('Сделки'),
        contacts.groupby('Contact Owner Name').size().rename('Контакты'),
        calls.groupby('Call Owner Name').size().rename('Звонки'),
    ], axis=1).fillna(0).sort_values(['Сделки', 'Контакты', 'Звонки'], ascending=False)
    # Один пузырь — одна запись; владельцы берутся из каждой таблицы отдельно.
    wheel_paid = (deals["Stage"] == "Payment Done") | deals["Initial Amount Paid"].gt(0)
    wheel_paid_contacts = paid_contacts
    wheel_layers = [
        ("Сделки · прочие", deals.loc[~wheel_paid], "Deal Owner Name", 0.04, 0.44, "#158BDF", 1.5),
        ("Сделки", deals.loc[wheel_paid], "Deal Owner Name", 0.04, 0.44, "#20B848", 8),
        ("Контакты", contacts, "Contact Owner Name", 0.46, 0.60, "#FF8A16", 6),
        ("Звонки", calls, "Call Owner Name", 0.62, 1.00, "#A33DD2", 5),
    ]
    wheel_owners = workload.index.tolist()
    wheel = go.Figure()
    wheel_rng = np.random.default_rng(42)
    owners = wheel_owners
    layer_x, layer_y, layer_data, layer_sizes = [], [], [], []
    for label, frame, owner_field, inner, outer, color, bubble_size in wheel_layers:
        xs, ys, info, sizes = [], [], [], []
        for j, owner in enumerate(owners):
            rows = frame.loc[(frame[owner_field] == owner)]
            radius = np.sqrt(wheel_rng.uniform(inner**2, outer**2, len(rows)))
            angle = (j + wheel_rng.uniform(0.04, 0.96, len(rows))) * 2*np.pi/len(owners) - np.pi/2
            xs.extend((radius*np.cos(angle)).round(5).tolist())
            ys.extend((radius*np.sin(angle)).round(5).tolist())
            if owner_field == "Contact Owner Name":
                linked_paid = rows["Id"].astype("string").str.strip().isin(wheel_paid_contacts)
            elif owner_field == "Call Owner Name":
                linked_paid = rows["CONTACTID"].astype("string").str.strip().isin(wheel_paid_contacts)
            else:
                linked_paid = (rows["Stage"] == "Payment Done") | rows["Initial Amount Paid"].gt(0)
            sizes.extend((np.where(linked_paid, bubble_size, 1.5)).tolist())
            info.extend([[owner, str(record), "Связано с оплатой" if linked else "Без подтверждённой связи с оплатой"]
                         for record, linked in zip(rows["Id"], linked_paid)])
        layer_x.append(xs); layer_y.append(ys); layer_data.append(info)
        layer_sizes.append(sizes)
    annotations, lines, label_points = [], [], []
    for j, owner in enumerate(owners):
        angle = (j+0.5)*2*np.pi/len(owners) - np.pi/2
        label_points.append((np.cos(angle), np.sin(angle), owner))
        edge = j*2*np.pi/len(owners) - np.pi/2
        lines.append(dict(type="line", x0=0.025*np.cos(edge), y0=0.025*np.sin(edge),
                          x1=np.cos(edge), y1=np.sin(edge), line=dict(color="#E5EAF2", width=0.7), layer="below"))
    for side in [-1, 1]:
        labels = sorted([point for point in label_points if (point[0] >= 0) == (side == 1)], key=lambda point: point[1])
        for (x, y, owner), label_y in zip(labels, np.linspace(-1.04, 1.04, len(labels))):
            annotations.append(dict(x=side*1.035, y=label_y, text=owner, showarrow=False,
                xanchor="left" if side == 1 else "right", font=dict(size=11, color="#36435B")))
            lines.append(dict(type="line", x0=x*1.02, y0=y*1.02, x1=side*1.025, y1=label_y,
                              line=dict(color="#CCD6E5", width=0.7), layer="below"))
    rings = [dict(type="circle", x0=-r, y0=-r, x1=r, y1=r,
                  line=dict(color="#DDE4EE", width=1), layer="below") for r in [0.45, 0.61, 1.015]]
    for i, (label, frame, owner_field, inner, outer, color, bubble_size) in enumerate(wheel_layers):
        wheel.add_trace(go.Scattergl(x=layer_x[i], y=layer_y[i], mode="markers", name=label, showlegend=(i != 0),
            customdata=layer_data[i], marker=dict(color=color, size=layer_sizes[i], opacity=0.45 if i == 0 else 0.85,
                line=dict(color="white", width=0 if i == 0 else 0.5)),
            hovertemplate="<b>%{customdata[0]}</b><br>"+label+" · ID %{customdata[1]}<br>%{customdata[2]}<extra></extra>"))
    wheel.update_layout(
        title=dict(text="Активность и результативность менеджеров", x=0.02, y=0.99, font=dict(size=25, color="#23324C")),
        template="plotly_white", height=950, autosize=True, margin=dict(l=15, r=15, t=155, b=25),
        font=dict(family="Arial", size=13, color="#36435B"),
        xaxis=dict(visible=False, range=[-1.52, 1.52], constrain="domain"),
        yaxis=dict(visible=False, range=[-1.06, 1.06], scaleanchor="x", scaleratio=1, constrain="domain"),
        annotations=annotations, shapes=rings+lines,
        legend=dict(orientation="h", x=1, xanchor="right", y=1.10, yanchor="top", itemsizing="constant"),
        hoverlabel=dict(bgcolor="white", font_color="#23324C", bordercolor="#D3DCEB"),
    )
    return wheel


app.layout = html.Div([
    dcc.Store(id='selected-city', data=None),
    html.H1('Анализ данных CRM онлайн-школы программирования X'),
    html.Div([
        html.Aside(filter_controls, className='filters'),
        html.Main([
            dcc.RadioItems(id='product', options=product_options, value=ALL, className='products'),
            html.Div(cards, className='metrics'),
            html.Div([
                html.Div([
                    html.H2('Города и немецкий язык'),
                    dcc.Loading(dcc.Graph(id='geo-map', responsive=True,
                        config={'displaylogo': False, 'displayModeBar': True, 'scrollZoom': True,
                                'modeBarButtons': [['zoomInMap', 'zoomOutMap', 'resetViewMap']]},
                        style={'height': '460px'}), color='#8373D7'),
                ], className='chart-panel', style={}),
                html.Div([
                    html.H2('Активность и результативность менеджеров'),
                    dcc.Loading(dcc.Graph(id='workload-wheel', responsive=True,
                        config={'displaylogo': False, 'displayModeBar': False},
                        style={'height': '460px'}), color='#8373D7'),
                ], className='chart-panel'),
            ], className='map-wheel-row'),
        ]),
    ], className='workspace'),
    html.Div([
        dcc.Tabs(id='animation-tabs', value='activity', className='animation-tabs', children=[
            dcc.Tab(label=label, value=value, className='animation-tab', selected_className='animation-tab animation-tab-selected')
            for label, value in [('Сделки и звонки', 'activity'), ('Сроки закрытия сделок', 'duration'),
                                  ('Количество звонков', 'calls'), ('Минуты звонков', 'minutes')]]
        ),
        dcc.Loading(dcc.Graph(id='daily-bubbles', responsive=True,
            config={'displaylogo': False, 'displayModeBar': False},
            style={'height': '650px'}), color='#8373D7'),
    ], className='chart-panel', style={'marginTop': '22px'}),
    html.Div([
        dcc.Loading(dcc.Graph(id='customer-journey', responsive=True,
            config={'displaylogo': False, 'displayModeBar': False},
            style={'height': '1700px'}), color='#8373D7'),
    ], className='chart-panel', style={'marginTop': '22px'}),
], className='dashboard')


@app.callback(
    Output('source', 'value'), Output('owner', 'value'), Output('result', 'value'),
    Output('german', 'value'), Output('product', 'value'),
    Input('reset', 'n_clicks'), prevent_initial_call=True,
)
def reset_filters(_):
    return (ALL,) * 5


@app.callback(
    Output('selected-city', 'data'),
    Input('geo-map', 'clickData'), Input('reset', 'n_clicks'),
    State('selected-city', 'data'), prevent_initial_call=True,
)
def select_city(click_data, reset_clicks, current_city):
    if ctx.triggered_id == 'reset':
        return None
    if click_data and click_data.get('points'):
        details = click_data['points'][0].get('customdata')
        if details:
            city = details[0]
            return None if city == current_city else city
    return no_update


@app.callback(
    Output('deals-value', 'children'), Output('paid-value', 'children'),
    Output('conversion-value', 'children'),
    Output('students-value', 'children'), Output('remaining-value', 'children'),
    Output('daily-bubbles', 'figure'),
    Output('daily-bubbles', 'style'),
    Input('source', 'value'), Input('owner', 'value'), Input('result', 'value'),
    Input('german', 'value'), Input('product', 'value'),
    Input('animation-tabs', 'value'),
    Input('selected-city', 'data'),
)
def update_dashboard(source, owner, result, german, product, animation_tab='activity', city=None):
    choices = [source, owner, result, german, product]
    selected = deals
    for field, value in zip(['Source', 'Deal Owner Name', 'Результат', 'German Level', 'Product'], choices):
        if value != ALL:
            selected = selected.loc[selected[field].eq(value)]
    if city is not None:
        selected = selected.loc[selected['City'].eq(city)]
    if animation_tab == 'activity':
        selected_calls = calls
        if city is not None or any(value != ALL for value in choices):
            selected_calls = calls.loc[calls['_contact'].isin(selected['_contact'].dropna().unique())]
        daily = pd.DataFrame({
            'Сделки': selected['_date'].value_counts().reindex(calendar, fill_value=0),
            'Звонки': selected_calls['_date'].value_counts().reindex(calendar, fill_value=0),
        })
        max_count = max(1, int(daily.to_numpy().max()))
        size_reference = 2 * max_count / 42**2
        frames = []
        for month in months:
            part = daily.loc[daily.index.strftime('%Y-%m') == month]
            day = part.index.day.to_numpy()
            traces = []
            for event in ['Сделки', 'Звонки']:
                counts = part[event].to_numpy()
                traces.append({
                    'type': 'scatter', 'mode': 'markers', 'name': event, 'legendgroup': event,
                    'x': day.tolist(), 'y': counts.tolist(),
                    'customdata': part.index.strftime('%d.%m.%Y').tolist(),
                    'marker': {'color': COLORS[event], 'size': counts.tolist(), 'sizemode': 'area',
                               'sizeref': size_reference, 'sizemin': 3, 'opacity': .72,
                               'line': {'color': '#ffffff', 'width': 1}},
                    'hovertemplate': '%{customdata}<br>' + event + ': %{y:,.0f}<extra></extra>',
                })
            for event in ['Сделки', 'Звонки']:
                counts = part[event].to_numpy()
                trend = np.polyval(np.polyfit(day, counts, 1), day) if len(day) > 1 else counts.astype(float)
                traces.append({
                    'type': 'scatter', 'mode': 'lines', 'name': event, 'legendgroup': event,
                    'showlegend': False, 'x': day.tolist(), 'y': trend.tolist(),
                    'line': {'color': COLORS[event], 'width': 2, 'dash': 'dot'},
                    'hovertemplate': 'Линейный тренд<br>' + event + ': %{y:.1f}<extra></extra>',
                })
            frames.append({'name': month, 'data': traces, 'traces': [0, 1, 2, 3]})
        figure = {
            'data': frames[0]['data'], 'frames': frames,
            'layout': {
                'autosize': True, 'height': 550, 'paper_bgcolor': '#ffffff', 'plot_bgcolor': '#ffffff',
                'font': {'family': 'Arial, sans-serif', 'size': 12, 'color': '#74747f'},
                'margin': {'l': 52, 'r': 20, 't': 42, 'b': 100},
                'legend': {'orientation': 'h', 'x': 0, 'y': 1.13},
                'xaxis': {'title': {'text': 'День месяца'}, 'range': [0, 32], 'dtick': 5,
                          'gridcolor': '#f2f1f6', 'zeroline': False, 'fixedrange': False},
                'yaxis': {'title': {'text': 'Количество'}, 'range': [0, max_count * 1.12],
                          'gridcolor': '#f2f1f6', 'zeroline': False},
                'sliders': [{'active': 0, 'x': .18, 'len': .82, 'y': -.15,
                    'currentvalue': {'prefix': '', 'font': {'size': 12}}, 'pad': {'t': 18},
                    'steps': [{'label': m, 'method': 'animate', 'args': [[m], {
                        'mode': 'immediate', 'frame': {'duration': 0, 'redraw': False},
                        'transition': {'duration': 0}}]} for m in months]}],
                'updatemenus': [{'type': 'buttons', 'direction': 'left', 'x': 0, 'y': -.15,
                    'xanchor': 'left', 'yanchor': 'top', 'pad': {'t': 18},
                    'bgcolor': '#ffffff', 'bordercolor': '#e4dfef',
                    'buttons': [
                        {'label': '▶', 'method': 'animate', 'args': [None, {'fromcurrent': True,
                            'frame': {'duration': 650, 'redraw': False}, 'transition': {'duration': 250}}]},
                        {'label': 'Ⅱ', 'method': 'animate', 'args': [[None], {'mode': 'immediate',
                            'frame': {'duration': 0, 'redraw': False}, 'transition': {'duration': 0}}]},
                    ]}],
            },
        }
        figure['layout']['height'] = 650
    else:
        figure = closed_animation(selected, animation_tab)
    graph_style = {'height': '650px'}
    if ctx.triggered_id == 'animation-tabs':
        return no_update, no_update, no_update, no_update, no_update, figure, graph_style
    count, paid_count = len(selected), int(selected['paid'].sum())
    conversion = f'{paid_count / count * 100:.2f}%'.replace('.', ',') if count else '-'
    students = selected.loc[selected['Offer Total Amount'].gt(0), 'Id'].nunique()
    remaining = (selected['Offer Total Amount'].sum(min_count=1)
        - selected['Initial Amount Paid'].sum(min_count=1)) if count else 0
    remaining_text = f'{remaining:,.0f} €'.replace(',', ' ').replace('.', ',') if pd.notna(remaining) else '-'
    return (f'{count:,}'.replace(',', ' '), f'{paid_count:,}'.replace(',', ' '), conversion,
            f'{students:,}'.replace(',', ' '), remaining_text, figure, graph_style)


@app.callback(
    Output('workload-wheel', 'figure'), Output('customer-journey', 'figure'),
    Input('source', 'value'), Input('owner', 'value'), Input('result', 'value'),
    Input('german', 'value'), Input('product', 'value'), Input('selected-city', 'data'),
)
def update_detail_charts(source, owner, result, german, product, city):
    choices = [source, owner, result, german, product]
    selected = deals
    for field, value in zip(['Source', 'Deal Owner Name', 'Результат', 'German Level', 'Product'], choices):
        if value != ALL:
            selected = selected.loc[selected[field].eq(value)]
    if city is not None:
        selected = selected.loc[selected['City'].eq(city)]
    wheel = workload_chart(selected, owner, city is not None or any(value != ALL for value in choices))
    wheel.update_layout(height=460, title_text='', margin=dict(l=0, r=0, t=36, b=0),
                        legend=dict(font=dict(size=9), x=0.02, xref='container', xanchor='left',
                                    y=1, yref='container', yanchor='top'), font=dict(size=9))
    for annotation in wheel.layout.annotations:
        annotation.font.size = 8
    return wheel, journey_chart(selected)


@app.callback(
    Output('geo-map', 'figure'),
    Input('source', 'value'), Input('owner', 'value'), Input('result', 'value'),
    Input('german', 'value'), Input('product', 'value'),
)
def update_map(source, owner, result, german, product):
    choices = [source, owner, result, german, product]
    selected = deals
    for field, value in zip(['Source', 'Deal Owner Name', 'Результат', 'German Level', 'Product'], choices):
        if value != ALL:
            selected = selected.loc[selected[field].eq(value)]
    known = selected.dropna(subset=['_lat', '_lon'])
    known = known.loc[known['German Level'].isin(['B1 и выше', 'ниже B1'])].copy()
    known['_b1'] = known['German Level'].eq('B1 и выше')
    known['_below'] = known['German Level'].eq('ниже B1')
    known['_paid_b1'] = known['paid'] & known['_b1']
    known['_paid_below'] = known['paid'] & known['_below']
    mapped = known.groupby(['City', '_lat', '_lon'], as_index=False, observed=True).agg(
        b1=('_b1', 'sum'), below=('_below', 'sum'),
        paid_b1=('_paid_b1', 'sum'), paid_below=('_paid_below', 'sum'))
    mapped['total'] = mapped['b1'] + mapped['below']
    max_total = max(1, int(mapped['total'].max())) if not mapped.empty else 1
    sector_features = {'B1+': [], 'хуже B1': []}
    hover_lat, hover_lon, hover_data = [], [], []
    earth_radius = 6378137.0
    for city, lat, lon, b1, below, paid_b1, paid_below, total in mapped.itertuples(index=False, name=None):
        center_x = earth_radius * np.radians(lon)
        center_y = earth_radius * np.log(np.tan(np.pi/4 + np.radians(np.clip(lat, -85.0511, 85.0511))/2))
        radius = max(22000.0, 120000.0 * np.sqrt(total / max_total))
        angle_start = np.pi / 2
        for label, amount, paid_amount in [('B1+', b1, paid_b1), ('хуже B1', below, paid_below)]:
            if amount == 0:
                continue
            fraction = amount / total
            angle_end = angle_start + 2 * np.pi * fraction
            angles = np.linspace(angle_start, angle_end, max(3, int(np.ceil(60*fraction)) + 1))
            x = center_x + radius * np.cos(angles)
            y = center_y + radius * np.sin(angles)
            arc = np.column_stack((np.degrees(x/earth_radius),
                np.degrees(2*np.arctan(np.exp(y/earth_radius)) - np.pi/2))).tolist()
            ring = arc + [arc[0]] if amount == total else [[lon, lat]] + arc + [[lon, lat]]
            sector_features[label].append({'type': 'Feature', 'properties': {},
                'geometry': {'type': 'Polygon', 'coordinates': [ring]}})
            middle = (angle_start + angle_end) / 2
            hover_lon.append(float(np.degrees((center_x + radius*.45*np.cos(middle))/earth_radius)))
            hover_lat.append(float(np.degrees(2*np.arctan(np.exp(
                (center_y + radius*.45*np.sin(middle))/earth_radius)) - np.pi/2)))
            hover_data.append([city, label, int(amount), int(paid_amount), float(fraction*100)])
            angle_start = angle_end
    map_layers, map_traces = [], []
    for label, color in [('B1+', '#55BFA8'), ('хуже B1', '#A43F50')]:
        map_layers.append({'sourcetype': 'geojson', 'type': 'fill', 'below': 'traces',
            'color': color, 'opacity': .85,
            'source': {'type': 'FeatureCollection', 'features': sector_features[label]}})
        map_traces.append({'type': 'scattermap', 'mode': 'markers', 'name': label,
            'lat': [None], 'lon': [None], 'marker': {'size': 10, 'color': color}, 'hoverinfo': 'skip'})
    map_traces.append({'type': 'scattermap', 'mode': 'markers', 'showlegend': False,
        'lat': hover_lat, 'lon': hover_lon, 'customdata': hover_data,
        'marker': {'size': 18, 'color': 'rgba(0,0,0,0)'},
        'hovertemplate': '<b>%{customdata[0]}</b><br>Немецкий: %{customdata[1]}'
            '<br>Сделки: %{customdata[2]} (%{customdata[4]:.1f}%)'
            '<br>Оплачено: %{customdata[3]}<extra></extra>'})
    geo_figure = {'data': map_traces, 'layout': {
        'height': 460, 'margin': {'l': 0, 'r': 0, 't': 36, 'b': 0},
        'font': {'family': 'Arial, sans-serif', 'size': 12, 'color': '#74747f'},
        'paper_bgcolor': '#ffffff', 'uirevision': 'crm-cities',
        'legend': {'orientation': 'h', 'x': 0.02, 'xref': 'container', 'xanchor': 'left',
                   'y': 1, 'yref': 'container', 'yanchor': 'top',
                   'itemclick': False, 'itemdoubleclick': False},
        'map': {'style': 'carto-positron', 'center': {'lat': 51.1, 'lon': 10.4}, 'zoom': 4.5,
            'layers': map_layers},
    }}
    geo_figure['layout']['datarevision'] = json.dumps(choices, ensure_ascii=False)
    if mapped.empty:
        geo_figure['data'] = []
        geo_figure['layout']['map']['layers'] = []
    return geo_figure
