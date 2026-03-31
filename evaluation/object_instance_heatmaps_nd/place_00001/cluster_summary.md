# Cluster Summary: place_00001

- group_type: place
- n_objects: 56
- n_clusters: 12

## Cluster 0
- candidate_instance_id: place:place_00001:cluster_000
- num_members: 4
- representative_label: picture frame
- representative_description: Square black frame with a wide white mat holding a sailboat print; glass produces a noticeable rectangular glare/reflection and the image is slightly blurred/low-resolution; frame appears wall-mounted and mostly visible in the crop; approximate distance ~3.0 m from camera.
- same_view_collision: True
- offending_view_ids: view_00004, view_00008
- member: object_id=19 view_id=view_00004 label=picture frame description=black square wall picture frame
- member: object_id=20 view_id=view_00004 label=picture frame description=white rectangular picture frame edge-crop
- member: object_id=47 view_id=view_00008 label=picture frame description=black-framed square wall picture
- member: object_id=48 view_id=view_00008 label=picture frame description=white-framed beach picture, partial

## Cluster 1
- candidate_instance_id: place:place_00001:cluster_001
- num_members: 5
- representative_label: window
- representative_description: object: window | attributes: grid panes, white frame, clear glass, outdoor view visible | camera_relation: distance=1.0, bearing=-35.0, laterality=left, verticality=middle | global_anchor: x=-12.0, z=-2.0 | surroundings: curtain | relation=adjacent | primary_dist=0.2 | global=(-11.5,-2.0); wall control (thermostat/keypad) | relation=to the right on same wall | primary_dist=0.6 | global=(-12.5,-2.5); decorative sign | relation=to the right and higher | primary_dist=1.0 | global=(-13.0,-3.0); right wall switch | relation=to the right | primary_dist=1.1 | global=(-13.0,-3.5); outdoor hose/wheel visible through glass | relation=outside, behind glass | primary_dist=3.5 | global=(-14.0,-0.5) | scene_context: floor_pattern=unknown; scene_attributes=beige painted wall, large grid-pane glass door/window on left, natural daylight coming from left, white electrical plates and controls on wall, coastal themed decorative sign; wall_color=beige
- same_view_collision: True
- offending_view_ids: view_00005, view_00009
- member: object_id=21 view_id=view_00005 label=blinds description=object: blinds | attrs: none | anchor: x=na, z=na | nearby: none
- member: object_id=22 view_id=view_00005 label=curtain description=object: curtain | attrs: pale blue, fabric, full-length | anchor: x=-11.5, z=-2.0 | nearby: window@(-12.0,-2.0)
- member: object_id=23 view_id=view_00005 label=window description=object: window | attrs: grid panes, white frame, clear glass, outdoor view visible | anchor: x=-12.0, z=-2.0 | nearby: curtain@(-11.5,-2.0), wall control (thermostat/keypad)@(-12.5,-2.5)
- member: object_id=49 view_id=view_00009 label=curtain description=object: curtain | attrs: blue-gray, fabric, hanging | anchor: x=-11.5, z=-2.0 | nearby: door@(-11.5,-2.0), light switch (left)@(-12.5,-2.5)
- member: object_id=50 view_id=view_00009 label=door description=object: door | attrs: white frame, multi-pane glass, exterior view visible through glass, attached to door frame | anchor: x=-11.5, z=-2.0 | nearby: curtain@(-11.5,-2.0), light switch (left)@(-12.5,-2.5)

## Cluster 2
- candidate_instance_id: place:place_00001:cluster_002
- num_members: 4
- representative_label: wall switch
- representative_description: object: wall switch | attributes: white, single rocker switch, standard size | camera_relation: distance=2.1, bearing=30.0, laterality=right, verticality=low | global_anchor: x=-13.0, z=-3.5 | surroundings: decorative sign | relation=above-left | primary_dist=0.6 | global=(-13.0,-3.0); wall control (thermostat/keypad) | relation=to the left | primary_dist=0.9 | global=(-12.5,-2.5); window | relation=left | primary_dist=2.0 | global=(-12.0,-2.0) | scene_context: floor_pattern=unknown; scene_attributes=beige painted wall, large grid-pane glass door/window on left, natural daylight coming from left, white electrical plates and controls on wall, coastal themed decorative sign; wall_color=beige
- same_view_collision: True
- offending_view_ids: view_00005, view_00009
- member: object_id=24 view_id=view_00005 label=other description=object: other | attrs: rectangular, white, likely thermostat or keypad | anchor: x=-12.5, z=-2.5 | nearby: window@(-12.0,-2.0), right wall switch@(-13.0,-3.5)
- member: object_id=26 view_id=view_00005 label=wall switch description=object: wall switch | attrs: white, single rocker switch, standard size | anchor: x=-13.0, z=-3.5 | nearby: decorative sign@(-13.0,-3.0), wall control (thermostat/keypad)@(-12.5,-2.5)
- member: object_id=51 view_id=view_00009 label=wall switch description=object: wall switch | attrs: white rocker, rectangular plate | anchor: x=-12.5, z=-2.5 | nearby: art / wall sign@(-13.0,-3.0), door@(-11.5,-2.0)
- member: object_id=53 view_id=view_00009 label=wall switch description=object: wall switch | attrs: white rocker, rectangular plate | anchor: x=-12.5, z=-3.5 | nearby: art / wall sign@(-13.0,-3.0), light switch (left)@(-12.5,-2.5)

## Cluster 3
- candidate_instance_id: place:place_00001:cluster_003
- num_members: 2
- representative_label: art
- representative_description: object: art | attributes: wooden, coastal theme, text reads 'Mermaids LIVE among us in VIRGINIA BEACH' (approx) | camera_relation: distance=2.0, bearing=20.0, laterality=right, verticality=high | global_anchor: x=-13.0, z=-3.0 | surroundings: right wall switch | relation=below | primary_dist=0.6 | global=(-13.0,-3.5); wall control (thermostat/keypad) | relation=to the left | primary_dist=1.2 | global=(-12.5,-2.5); window | relation=to the left and lower | primary_dist=1.8 | global=(-12.0,-2.0) | scene_context: floor_pattern=unknown; scene_attributes=beige painted wall, large grid-pane glass door/window on left, natural daylight coming from left, white electrical plates and controls on wall, coastal themed decorative sign; wall_color=beige
- same_view_collision: False
- member: object_id=25 view_id=view_00005 label=art description=object: art | attrs: wooden, coastal theme, text reads 'Mermaids LIVE among us in VIRGINIA BEACH' (approx) | anchor: x=-13.0, z=-3.0 | nearby: right wall switch@(-13.0,-3.5), wall control (thermostat/keypad)@(-12.5,-2.5)
- member: object_id=52 view_id=view_00009 label=art description=object: art | attrs: wooden plank style, white background, blue lettering, weathered look | anchor: x=-12.5, z=-3.5 | nearby: light switch (left)@(-12.5,-2.5), light switch (right)@(-12.5,-3.5)

## Cluster 4
- candidate_instance_id: place:place_00001:cluster_004
- num_members: 6
- representative_label: couch
- representative_description: Close-up of a dark glossy leather three-seat couch with padded, vertically segmented back cushions and stitched seat panels; right armrest and the right portion of the couch are visible and the object is partially cropped at the image edge, estimated about 0.8 m from the camera.
- same_view_collision: True
- offending_view_ids: view_00007, view_00011
- member: object_id=27 view_id=view_00006 label=couch description=dark brown leather couch, partial
- member: object_id=35 view_id=view_00007 label=couch description=black leather three-seat couch edge crop
- member: object_id=36 view_id=view_00007 label=couch description=dark brown leather recliner, front
- member: object_id=54 view_id=view_00010 label=couch description=dark brown leather loveseat, cropped
- member: object_id=62 view_id=view_00011 label=couch description=dark glossy leather couch, partial
- member: object_id=63 view_id=view_00011 label=couch description=dark brown leather recliner front crop

## Cluster 5
- candidate_instance_id: place:place_00001:cluster_005
- num_members: 17
- representative_label: cabinet
- representative_description: Close-up of a lower dark-brown stained wooden cabinet door with a recessed rectangular center panel and raised molding; glossy varnish with visible scuffs and small chips at the edges, right side of the door is cropped by the image; appears to be a lower cabinet near the floor, approximately 0.5 m from the camera.
- same_view_collision: True
- offending_view_ids: view_00006, view_00007, view_00010, view_00011
- member: object_id=28 view_id=view_00006 label=side table description=dark brown wooden side table
- member: object_id=29 view_id=view_00006 label=rug description=light beige woven rug edge crop
- member: object_id=31 view_id=view_00006 label=rug description=beige low-pile rug edge crop
- member: object_id=33 view_id=view_00006 label=rug description=dark navy rectangular rug, edge-cropped
- member: object_id=34 view_id=view_00006 label=side table description=dark brown varnished side table edge
- member: object_id=39 view_id=view_00007 label=cabinet description=dark brown upper cabinet panel
- member: object_id=41 view_id=view_00007 label=dining table description=dark wooden dining table edge
- member: object_id=45 view_id=view_00007 label=cabinet description=dark brown cabinet door edge-cropped
- member: object_id=55 view_id=view_00010 label=side table description=dark wooden side table, front-facing
- member: object_id=56 view_id=view_00010 label=rug description=light beige low-pile rug edge-cropped
- member: object_id=58 view_id=view_00010 label=rug description=beige low-pile rug edge crop
- member: object_id=60 view_id=view_00010 label=rug description=dark rectangular floor rug, edge-cropped
- member: object_id=61 view_id=view_00010 label=side table description=dark brown glossy side table edge
- member: object_id=65 view_id=view_00011 label=cabinet description=dark wood cabinet upper doors cropped
- member: object_id=67 view_id=view_00011 label=cabinet description=tall dark wooden cabinet door
- member: object_id=72 view_id=view_00011 label=cabinet description=dark wooden cabinet lower panel (partial)
- member: object_id=74 view_id=view_00011 label=cabinet description=dark wooden cabinet lower section

## Cluster 6
- candidate_instance_id: place:place_00001:cluster_006
- num_members: 2
- representative_label: fireplace
- representative_description: Rectangular black metal fireplace insert with a glass front showing a dark, unlit interior and a single soot-stained log or ceramic log; glass is slightly reflective and the metal frame shows minor wear and ash residue, visible from roughly 1.5 meters away.
- same_view_collision: False
- member: object_id=30 view_id=view_00006 label=fireplace description=black metal fireplace insert
- member: object_id=57 view_id=view_00010 label=fireplace description=black metal fireplace insert

## Cluster 7
- candidate_instance_id: place:place_00001:cluster_007
- num_members: 2
- representative_label: window
- representative_description: Large dark rectangular reflective window showing a dim interior reflection of a staircase, table lamp, and furniture; glass appears slightly blurred or smudged and the crop is edge-cut so the frame is partially visible — approximate distance ~2.0 m from camera.
- same_view_collision: False
- member: object_id=32 view_id=view_00006 label=window description=dark reflective window, edge-cropped
- member: object_id=59 view_id=view_00010 label=window description=dark reflective glass window edge-cropped

## Cluster 8
- candidate_instance_id: place:place_00001:cluster_008
- num_members: 2
- representative_label: table lamp
- representative_description: Small white tabletop lamp with a scalloped fabric lampshade and visible stitched trim, white/ivory textured ceramic base with decorative detailing and a turquoise object at the base; lower-left of the lamp is cropped/partial in the image; estimated height ~30–45 cm and approximately 0.8 m from the camera.
- same_view_collision: False
- member: object_id=37 view_id=view_00007 label=table lamp description=white scalloped shade table lamp
- member: object_id=64 view_id=view_00011 label=table lamp description=white pleated shade table lamp (partial)

## Cluster 9
- candidate_instance_id: place:place_00001:cluster_009
- num_members: 6
- representative_label: chair
- representative_description: Ornate dark reddish-brown lacquered wooden chair with a pierced, carved backrest, curved front legs and a visible front stretcher; patterned upholstered seat is partially visible and the sides are slightly edge-cropped in the narrow vertical crop — appears to be photographed at approximately 1.5 meters distance.
- same_view_collision: True
- offending_view_ids: view_00007, view_00011
- member: object_id=38 view_id=view_00007 label=chair description=dark wooden chair partial crop
- member: object_id=40 view_id=view_00007 label=chair description=dark reddish-brown carved wooden chair
- member: object_id=46 view_id=view_00007 label=chair description=dark wooden chair, edge-cropped
- member: object_id=66 view_id=view_00011 label=chair description=ornate dark-red wooden chair edge-cropped
- member: object_id=70 view_id=view_00011 label=chair description=dark carved wooden chair edge
- member: object_id=71 view_id=view_00011 label=chair description=dark wooden chair edge crop

## Cluster 10
- candidate_instance_id: place:place_00001:cluster_010
- num_members: 4
- representative_label: microwave
- representative_description: Close frontal view of a stainless steel countertop microwave showing a reflective dark glass door with two visible interior light reflections, a vertical black control panel on the right and a horizontal door handle; the appliance fills most of the crop and appears close to the camera (approximately 0.6 m).
- same_view_collision: True
- offending_view_ids: view_00007, view_00011
- member: object_id=42 view_id=view_00007 label=microwave description=stainless steel microwave front, black door
- member: object_id=43 view_id=view_00007 label=oven description=stainless-steel oven front, towel
- member: object_id=69 view_id=view_00011 label=oven description=stainless-steel oven front with towel
- member: object_id=73 view_id=view_00011 label=microwave description=stainless steel microwave front crop

## Cluster 11
- candidate_instance_id: place:place_00001:cluster_011
- num_members: 2
- representative_label: refrigerator
- representative_description: Full-height stainless steel double-door refrigerator with vertical center handles and reflective metallic surface showing smudges and faint reflections; occupies most of the crop and is slightly edge-cropped at the sides, estimated distance ≈ 1.5 m from camera.
- same_view_collision: False
- member: object_id=44 view_id=view_00007 label=refrigerator description=stainless steel double-door refrigerator
- member: object_id=68 view_id=view_00011 label=refrigerator description=stainless steel double-door refrigerator
