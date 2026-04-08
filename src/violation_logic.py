def check_violations(detections, violation_thresh=0.85):
    """
    Business logic for identifying violations.

    Custom model mode (Helmet / No-Helmet / License Plate classes available):
        Flags a motorcycle whose rider has a 'No-Helmet' detection + an associated License Plate.

    COCO model mode (person / motorcycle only):
        Flags a motorcycle that has a nearby rider (person) as a *potential* violation
        since COCO cannot distinguish helmeted vs. un-helmeted riders.
        No OCR is attempted (no license plate class in COCO).
    """

    motorcycles  = [d for d in detections if d['class_name'] == 'Motorcycle']
    no_helmets   = [d for d in detections if d['class_name'] == 'No-Helmet']
    helmets      = [d for d in detections if d['class_name'] == 'Helmet']
    plates       = [d for d in detections if d['class_name'] == 'License Plate']
    riders       = [d for d in detections if d['class_name'] == 'Rider']   # COCO mode

    violations = []
    
    PLATE_IOU_THRESH = 0.60 # >60% IoU overlap with the motorcycle box

    # ── Custom model mode ───────────────────────────────────────────────────
    if no_helmets or helmets or plates:
        for mc in motorcycles:
            mc_box = mc['box']

            assigned_no_helmet = None
            for nh in no_helmets:
                if nh['conf'] > violation_thresh and overlaps(nh['box'], mc_box):
                    assigned_no_helmet = nh
                    break
                    
            # Ambiguity Handling: check if helmet is being carried but not worn
            # If helmet is detected, check if its y-coordinate is roughly in the top 30% of the rider's bbox.
            # Since we only have motorcycle bbox directly, we rough estimate it based on motorcycle bbox / or assume helmet is high up.
            # For a more robust approach, we need the "rider" bbox. Let's use the topmost part of the MC box as a proxy if rider not present.
            carried_helmet = None
            for h in helmets:
                h_box = h['box']
                if overlaps(h_box, mc_box):
                    # Helmet bounding box center Y
                    h_center_y = (h_box[1] + h_box[3]) / 2.0
                    # MC bounding box center Y
                    mc_center_y = (mc_box[1] + mc_box[3]) / 2.0
                    
                    # If the helmet is in the bottom half of the motorcycle detection, it's likely being carried.
                    if h_center_y > mc_center_y:
                         carried_helmet = h
                         break

            # If a helmet is carried (or there is a clear no-helmet detection), flag violation.
            if assigned_no_helmet or carried_helmet:
                assigned_plate = None
                for p in plates:
                    # Spatial Analysis: Logic to ensure the license plate bounding box has a >60% IoU overlap with the motorcycle box.
                    if overlaps(p['box'], mc_box, iou_threshold=PLATE_IOU_THRESH):
                        assigned_plate = p
                        break

                if assigned_plate:
                    violations.append({
                        'type':       'no_helmet' if assigned_no_helmet else 'helmet_carried_not_worn',
                        'motorcycle': mc,
                        'no_helmet':  assigned_no_helmet or carried_helmet,
                        'plate':      assigned_plate,
                    })

    # ── COCO model mode (approximate) ──────────────────────────────────────
    else:
        for mc in motorcycles:
            mc_box = mc['box']
            for rider in riders:
                if overlaps(rider['box'], mc_box, iou_threshold=0.05):
                    violations.append({
                        'type':       'potential_rider',
                        'motorcycle': mc,
                        'rider':      rider,
                        'plate':      None,   # COCO has no license plate class
                    })
                    break

    return violations


# ── Geometry helpers ────────────────────────────────────────────────────────

def is_inside(inner_box, outer_box):
    """Return True when inner_box is fully contained within outer_box."""
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2


def overlaps(box_a, box_b, iou_threshold=0.0):
    """
    Return True when box_a and box_b have any overlap (iou_threshold=0)
    or when their IoU exceeds the given threshold.
    Also returns True when one box is fully inside the other.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return False
    if iou_threshold == 0.0:
        return True

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    iou = inter_area / (area_a + area_b - inter_area + 1e-6)
    return iou >= iou_threshold
