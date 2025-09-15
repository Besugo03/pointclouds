import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import re
import sys

def parse_point_cloud_file(file_path):
    """
    Analizza un file di nuvola di punti, estraendo punti, normali e curvature.
    """
    surfaces = []
    current_surface_data = []
    surface_name = "Sconosciuta"
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                header_match = re.match(r'\[\d+\]\s+SRF\s*:\s*(.*)', line)
                if header_match:
                    if current_surface_data:
                        data_array = np.array(current_surface_data, dtype=np.float64)
                        surfaces.append({
                            "name": surface_name,
                            "points": data_array[:, :3],
                            "normals": data_array[:, 3:6],
                            "curvatures": data_array[:, 6]
                        })
                        current_surface_data = []
                    surface_name = header_match.group(1).strip()
                else:
                    try:
                        parts = [float(p) for p in line.split(',')]
                        if len(parts) == 7:
                            current_surface_data.append(parts)
                    except ValueError:
                        print(f"Attenzione: Riga non valida saltata: '{line}'", file=sys.stderr)
            if current_surface_data:
                data_array = np.array(current_surface_data, dtype=np.float64)
                surfaces.append({
                    "name": surface_name,
                    "points": data_array[:, :3],
                    "normals": data_array[:, 3:6],
                    "curvatures": data_array[:, 6]
                })
    except FileNotFoundError:
        print(f"Errore: Il file '{file_path}' non è stato trovato.", file=sys.stderr)
        return None
    return surfaces

def visualize_and_save_surfaces(
    surfaces_data, 
    color_mode='surface', 
    output_filename="visualizzazione.html",
    show_normals=False,
    normal_subsample=200,
    normal_scale=1.0
):
    """
    Crea una visualizzazione 3D con Plotly, con opzioni per visualizzare le normali.
    """
    if not surfaces_data:
        print("Nessun dato da visualizzare.", file=sys.stderr)
        return

    print(f"\nCreazione della figura 3D (modalità: {color_mode}, normali: {show_normals})...", file=sys.stderr)
    fig = go.Figure()

    # Logica di colorazione (invariata)
    cmin, cmax = (None, None)
    if color_mode == 'curvature':
        all_curvatures = np.concatenate([s['curvatures'] for s in surfaces_data])
        cmin, cmax = all_curvatures.min(), all_curvatures.max()

    cmap = plt.get_cmap('gist_rainbow', len(surfaces_data))
    colormaps_for_gradients = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis'] 
    
    for i, surface in enumerate(surfaces_data):
        # ... (Codice per definire marker_properties in base a color_mode - invariato) ...
        marker_properties = {}
        if color_mode == 'surface':
            color_rgb = cmap(i)[:3]
            marker_properties = dict(size=1.5, color=f'rgb({int(color_rgb[0]*255)}, {int(color_rgb[1]*255)}, {int(color_rgb[2]*255)})')
        elif color_mode == 'curvature':
            marker_properties = dict(size=1.5, color=surface["curvatures"], colorscale='Viridis', cmin=cmin, cmax=cmax, showscale=(i == 0), colorbar=dict(title='Curvatura Globale') if i == 0 else None)
        elif color_mode == 'curvature_by_surface':
            marker_properties = dict(size=1.5, color=surface["curvatures"], colorscale=colormaps_for_gradients[i % len(colormaps_for_gradients)], showscale=True, colorbar=dict(title=f'Curvatura {surface["name"]}'))

        # Aggiungi la traccia per i punti
        fig.add_trace(go.Scatter3d(
            x=surface["points"][:, 0], y=surface["points"][:, 1], z=surface["points"][:, 2],
            mode='markers', name=surface["name"], marker=marker_properties
        ))
        print(f" - Aggiunta superficie '{surface['name']}'", file=sys.stderr)

        # --- NUOVA SEZIONE PER LE NORMALI ---
        if show_normals:
            points = surface['points']
            normals = surface['normals']
            num_points = len(points)
            
            # Seleziona un sottoinsieme casuale di indici
            if num_points > normal_subsample:
                indices = np.random.choice(num_points, size=normal_subsample, replace=False)
            else:
                indices = np.arange(num_points)

            # Prepara le coordinate per le linee delle normali
            lines_x, lines_y, lines_z = [], [], []
            for idx in indices:
                p = points[idx]
                n = normals[idx]
                end_point = p + n * normal_scale
                
                lines_x.extend([p[0], end_point[0], None]) # 'None' serve a spezzare la linea
                lines_y.extend([p[1], end_point[1], None])
                lines_z.extend([p[2], end_point[2], None])
            
            # Aggiungi una traccia per le normali della superficie corrente
            fig.add_trace(go.Scatter3d(
                x=lines_x, y=lines_y, z=lines_z,
                mode='lines',
                name=f'Normali {surface["name"]}',
                line=dict(color='black', width=3),
                showlegend=True # Mostra la legenda per le normali di ogni superficie
            ))

    fig.update_layout(title=f"Visualizzatore Nuvole di Punti - Modalità: {color_mode}",
                      scene=dict(xaxis_title='Asse X', yaxis_title='Asse Y', zaxis_title='Asse Z',
                                 aspectratio=dict(x=1, y=1, z=1)))
    
    fig.write_html(output_filename)
    print(f"Successo! Visualizzazione salvata in: {output_filename}", file=sys.stderr)

# --- Esecuzione Principale ---
if __name__ == "__main__":
    file_da_aprire = "/home/besugo/Downloads/MODEL_param/extracted_model/MODEL_param/Pipes_1000_processed/P_03.28.25_0184_pts_processed.txt"
    
    print(f"Tentativo di aprire e analizzare il file: '{file_da_aprire}'", file=sys.stderr)
    dati_superfici = parse_point_cloud_file(file_da_aprire)
    
    if dati_superfici:
        # 1. Visualizzazione per superficie (senza normali)
        visualize_and_save_surfaces(dati_superfici, color_mode='surface', output_filename='vis_per_superficie.html')
        
        # 2. Visualizzazione per curvatura globale (senza normali)
        visualize_and_save_surfaces(dati_superfici, color_mode='curvature', output_filename='vis_per_curvatura_globale.html')
        
        # 3. Visualizzazione con curvatura per superficie (senza normali)
        visualize_and_save_surfaces(dati_superfici, color_mode='curvature_by_surface', output_filename='vis_curvatura_per_superficie.html')

        # 4. NUOVA: Visualizzazione per superficie CON NORMALI
        visualize_and_save_surfaces(dati_superfici, 
                                    color_mode='surface', 
                                    output_filename='vis_con_normali.html',
                                    show_normals=True,
                                    normal_subsample=750, # Mostra n normali per superficie
                                    normal_scale=2.5)     # Lunghezza delle normali (da aggiustare)