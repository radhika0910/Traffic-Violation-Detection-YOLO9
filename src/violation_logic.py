def check_violations(detections, violation_thresh=0.85):
    """
    Business logic for identifying violations.
    - Custom mode: Checks for 'No-Helmet' or missing 'Helmet' box + Plate.
    - COCO mode: Flags 'Rider' + 'Motorcycle' if no helmet class is found.
    """

    motorcycles  = [d for d in detections if d['class_name'].lower() == 'motorcycle']
    no_helmets   = [d for d in detections if d['class_name'].lower() in ['no-helmet', 'no_helmet']]
    helmets      = [d for d in detections if d['class_name'].lower() == 'helmet']
    plates       = [d for d in detections if d['class_name'].lower() in ['license plate', 'plate', 'license_plate']]
    riders       = [d for d in detections if d['class_name'].lower() in ['rider', 'person']]

    violations = []
    
    PLATE_IOU_THRESH = 0.50 

    for mc in motorcycles:
        mc_box = mc['box']
        
        # 1. Look for associated riders
        assigned_riders = [r for r in riders if overlaps(r['box'], mc_box, iou_threshold=0.05)]
        if not assigned_riders:
            continue
            
        # Check for Triple Riding (3 or more people on one motorcycle)
        is_triple_riding = len(assigned_riders) >= 3
        
        for rider in assigned_riders:
            rider_box = rider['box']
            
            # --- CUSTOM MODEL LOGIC ---
            # If we see a 'Helmet' detection on this rider, skip helmet violation 
            # (but keep triple riding violation if applicable)
            has_helmet = any(overlaps(h['box'], rider_box, iou_threshold=0.2) for h in helmets)
            
            # If rider has helmet AND it's not triple riding, no violation for this rider
            if has_helmet and not is_triple_riding:
                continue
                
            # If we see a 'No-Helmet' detection or have no helmet info at all (COCO mode)
            is_definitely_no_helmet = any(overlaps(nh['box'], rider_box, iou_threshold=0.2) for nh in no_helmets)
            
            # 2. Look for associated plate
            assigned_plate = None
            for p in plates:
                if overlaps(p['box'], mc_box, iou_threshold=PLATE_IOU_THRESH):
                    assigned_plate = p
                    break

            # 3. Determine violation type
            reasons = []
            if is_triple_riding:
                reasons.append("Triple Riding")
            
            if not has_helmet:
                if is_definitely_no_helmet:
                    reasons.append("No Helmet Detected")
                elif not any(helmets) and not any(no_helmets):
                    # COCO Mode: Infer violation because we can't see helmet class
                    reasons.append("Potential Helmet Violation")
                else:
                    # Custom model but no helmet box found
                    reasons.append("Missing Helmet")

            if not reasons:
                continue

            viol_type = " & ".join(reasons)

            # Always return the motorcycle box as a anchor if plate is missing
            # The app will use this to "hunt" for the plate via OCR
            violations.append({
                'type':       viol_type,
                'motorcycle': mc,
                'rider':      rider,
                'plate':      assigned_plate,
            })

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
        
    # Check if one is entirely inside the other (subset)
    if is_inside(box_a, box_b) or is_inside(box_b, box_a):
        return True
        
    if iou_threshold == 0.0:
        return True

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    iou = inter_area / (area_a + area_b - inter_area + 1e-6)
    return iou >= iou_threshold
