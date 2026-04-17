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
        
        # Look for associated plate
        assigned_plate = None
        for p in plates:
            if overlaps(p['box'], mc_box, iou_threshold=PLATE_IOU_THRESH):
                assigned_plate = p
                break

        # Look for associated riders
        assigned_riders = [r for r in riders if overlaps(r['box'], mc_box, iou_threshold=0.05)]
        
        if not assigned_riders:
            violations.append({
                'type':       'Vehicle Detected',
                'motorcycle': mc,
                'rider':      None,
                'plate':      assigned_plate,
            })
            continue
            
        # Check for Triple Riding (3 or more people on one motorcycle)
        is_triple_riding = len(assigned_riders) >= 3
        
        mc_reasons = set()
        if is_triple_riding:
            mc_reasons.add("Triple Riding")
        
        for rider in assigned_riders:
            rx1, ry1, rx2, ry2 = rider['box']
            # COCO Person bounding box sometimes misses the head, especially for pillion riders.
            # Expand the top of the box to safely catch the helmet.
            h_rider = ry2 - ry1
            expanded_rider_box = (rx1, max(0, ry1 - int(h_rider * 0.4)), rx2, ry2)
            
            my_helmets = [h for h in helmets if overlaps(h['box'], expanded_rider_box, iou_threshold=0.1, use_min_area=True)]
            my_no_helmets = [nh for nh in no_helmets if overlaps(nh['box'], expanded_rider_box, iou_threshold=0.1, use_min_area=True)]
            
            best_helmet_conf = max([h['conf'] for h in my_helmets] + [0.0])
            best_no_helmet_conf = max([nh['conf'] for nh in my_no_helmets] + [0.0])
            
            has_helmet = best_helmet_conf > best_no_helmet_conf and best_helmet_conf > 0.0
            is_definitely_no_helmet = best_no_helmet_conf >= best_helmet_conf and best_no_helmet_conf > 0.0
            
            if not has_helmet:
                if is_definitely_no_helmet:
                    mc_reasons.add("No Helmet Detected")
                elif not any(helmets) and not any(no_helmets):
                    # COCO Mode: Infer violation because we can't see helmet class
                    mc_reasons.add("Potential Helmet Violation")
                else:
                    # Custom model but no helmet box found
                    mc_reasons.add("Missing Helmet")

        if not mc_reasons:
            viol_type = "Vehicle Detected"
        else:
            viol_type = " & ".join(sorted(list(mc_reasons)))

        violations.append({
            'type':       viol_type,
            'motorcycle': mc,
            'rider':      assigned_riders[0] if assigned_riders else None,
            'plate':      assigned_plate,
        })

    return violations


# ── Geometry helpers ────────────────────────────────────────────────────────

def is_inside(inner_box, outer_box):
    """Return True when inner_box is fully contained within outer_box."""
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2


def overlaps(box_a, box_b, iou_threshold=0.0, use_min_area=False):
    """
    Return True when box_a and box_b have any overlap (iou_threshold=0)
    or when their IoU (or IoA if use_min_area=True) exceeds the given threshold.
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
    
    if use_min_area:
        iou = inter_area / (min(area_a, area_b) + 1e-6)
    else:
        iou = inter_area / (area_a + area_b - inter_area + 1e-6)
        
    return iou >= iou_threshold
