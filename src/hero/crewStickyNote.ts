import type { AstronautPoseId, CrewAstronautSpec } from './crewAstronaut.ts'

/** Draw a Content-Warning-inspired sticky-note face (or photo when provided) onto a 512x512 canvas. */
export async function renderStickyNoteCanvas(spec: CrewAstronautSpec): Promise<HTMLCanvasElement> {
  const size = 512
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!

  const grad = ctx.createLinearGradient(0, 0, size, size)
  grad.addColorStop(0, '#f8e99b')
  grad.addColorStop(0.65, '#f2db78')
  grad.addColorStop(1, '#dfc25c')
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, size, size)

  ctx.fillStyle = 'rgba(130, 100, 30, 0.09)'
  ctx.fillRect(0, 0, size, 48)

  if (spec.photo) {
    try {
      const img = await new Promise<HTMLImageElement>((resolve, reject) => {
        const el = new Image()
        el.crossOrigin = 'anonymous'
        el.onload = () => resolve(el)
        el.onerror = reject
        el.src = spec.photo!
      })
      const pad = 34
      const boxW = size - pad * 2
      const boxH = size - pad * 2 - 54
      ctx.save()
      ctx.beginPath()
      ctx.rect(pad, pad, boxW, boxH)
      ctx.clip()
      const scaleCover = Math.max(boxW / img.width, boxH / img.height)
      const dw = img.width * scaleCover
      const dh = img.height * scaleCover
      ctx.drawImage(img, pad + (boxW - dw) / 2, pad + (boxH - dh) / 2, dw, dh)
      ctx.restore()
      ctx.strokeStyle = 'rgba(40, 32, 18, 0.25)'
      ctx.lineWidth = 4
      ctx.strokeRect(pad, pad, boxW, boxH)
    } catch {
      drawPanicDoodle(ctx, spec.pose, size)
    }
  } else {
    drawPanicDoodle(ctx, spec.pose, size)
  }

  ctx.fillStyle = '#1b1814'
  ctx.font = '600 40px "DM Sans", system-ui, sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText(spec.name, size / 2, size - 46)

  return canvas
}

function drawPanicDoodle(ctx: CanvasRenderingContext2D, pose: AstronautPoseId, size: number) {
  ctx.strokeStyle = '#171512'
  ctx.fillStyle = '#171512'
  ctx.lineWidth = 18
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'

  const cx = size / 2
  const cy = size * 0.44

  if (pose === 'cling') {
    ctx.beginPath()
    ctx.arc(cx - 82, cy - 34, 42, 0, Math.PI * 2)
    ctx.stroke()
    ctx.beginPath()
    ctx.arc(cx + 82, cy - 34, 42, 0, Math.PI * 2)
    ctx.stroke()
    ctx.beginPath()
    ctx.arc(cx - 76, cy - 42, 14, 0, Math.PI * 2)
    ctx.fill()
    ctx.beginPath()
    ctx.arc(cx + 76, cy - 42, 14, 0, Math.PI * 2)
    ctx.fill()
    ctx.beginPath()
    ctx.ellipse(cx, cy + 68, 38, 52, 0, 0, Math.PI * 2)
    ctx.stroke()
    ctx.fillStyle = '#3b82c4'
    ctx.beginPath()
    ctx.moveTo(cx + 158, cy - 115)
    ctx.quadraticCurveTo(cx + 186, cy - 64, cx + 158, cy - 50)
    ctx.quadraticCurveTo(cx + 130, cy - 64, cx + 158, cy - 115)
    ctx.fill()
  } else if (pose === 'laptop') {
    ctx.beginPath()
    ctx.moveTo(cx - 126, cy - 82)
    ctx.lineTo(cx - 40, cy - 48)
    ctx.moveTo(cx + 126, cy - 82)
    ctx.lineTo(cx + 40, cy - 48)
    ctx.stroke()
    ctx.beginPath()
    ctx.arc(cx - 76, cy - 12, 18, 0, Math.PI * 2)
    ctx.fill()
    ctx.beginPath()
    ctx.arc(cx + 76, cy - 12, 18, 0, Math.PI * 2)
    ctx.fill()
    ctx.beginPath()
    ctx.ellipse(cx, cy + 72, 54, 40, 0, 0, Math.PI * 2)
    ctx.stroke()
  } else if (pose === 'brace') {
    ctx.beginPath()
    ctx.moveTo(cx - 118, cy - 64)
    ctx.lineTo(cx - 48, cy - 26)
    ctx.lineTo(cx - 118, cy + 12)
    ctx.moveTo(cx + 118, cy - 64)
    ctx.lineTo(cx + 48, cy - 26)
    ctx.lineTo(cx + 118, cy + 12)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(cx - 92, cy + 76)
    ctx.quadraticCurveTo(cx - 46, cy + 44, cx, cy + 76)
    ctx.quadraticCurveTo(cx + 46, cy + 108, cx + 92, cy + 76)
    ctx.stroke()
  } else if (pose === 'hang') {
    ctx.beginPath()
    ctx.moveTo(cx - 124, cy - 44)
    ctx.lineTo(cx - 34, cy - 44)
    ctx.moveTo(cx - 79, cy - 44)
    ctx.lineTo(cx - 79, cy + 44)
    ctx.moveTo(cx + 34, cy - 44)
    ctx.lineTo(cx + 124, cy - 44)
    ctx.moveTo(cx + 79, cy - 44)
    ctx.lineTo(cx + 79, cy + 44)
    ctx.stroke()
    ctx.beginPath()
    ctx.arc(cx, cy + 92, 34, Math.PI * 1.1, Math.PI * 1.9)
    ctx.stroke()
  } else if (pose === 'tumble') {
    const drawSpiral = (sx: number, sy: number) => {
      ctx.beginPath()
      for (let i = 0; i <= 54; i++) {
        const a = i * 0.34
        const r = i * 0.92
        const px = sx + Math.cos(a) * r
        const py = sy + Math.sin(a) * r
        if (i === 0) ctx.moveTo(px, py)
        else ctx.lineTo(px, py)
      }
      ctx.stroke()
    }
    ctx.lineWidth = 14
    drawSpiral(cx - 82, cy - 26)
    drawSpiral(cx + 82, cy - 26)
    ctx.lineWidth = 16
    ctx.beginPath()
    ctx.arc(cx + 12, cy + 82, 26, 0, Math.PI * 2)
    ctx.stroke()
  } else {
    ctx.beginPath()
    ctx.arc(cx - 84, cy - 24, 18, 0, Math.PI * 2)
    ctx.fill()
    ctx.beginPath()
    ctx.arc(cx + 84, cy - 24, 18, 0, Math.PI * 2)
    ctx.fill()
    ctx.beginPath()
    ctx.moveTo(cx - 64, cy + 62)
    ctx.lineTo(cx + 64, cy + 62)
    ctx.stroke()
  }
}
