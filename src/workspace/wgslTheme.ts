import { ExtensionHostKind, registerExtension } from '@codingame/monaco-vscode-api/extensions'

export const wgslLanguageId = 'earth-wgsl'
export const wgslThemeName = 'Earth GitHub OLED'
export const wgslKeywords = 'alias break case const const_assert continue continuing default diagnostic discard else enable false fn for if let loop override requires return struct switch true var while'.split(' ')
export const wgslTypes = 'array atomic bool f16 f32 i32 mat2x2f mat3x3f mat4x4f ptr sampler u32 vec2 vec3 vec4 vec2f vec3f vec4f vec2i vec3i vec4i vec2u vec3u vec4u'.split(' ')

const theme = {
  type: 'dark',
  tokenColors: [
    { scope: 'comment.wgsl', settings: { foreground: '#929292' } },
    { scope: 'keyword.wgsl', settings: { foreground: '#ff7b72' } },
    { scope: 'storage.type.wgsl', settings: { foreground: '#ffa657' } },
    { scope: 'entity.name.function.wgsl', settings: { foreground: '#d2a8ff' } },
    { scope: 'constant.numeric.wgsl', settings: { foreground: '#79c0ff' } },
    { scope: 'constant.language.wgsl', settings: { foreground: '#79c0ff' } },
    { scope: 'meta.annotation.wgsl', settings: { foreground: '#7ee787' } },
    { scope: 'variable.other.identifier.wgsl', settings: { foreground: '#c9d1d9' } },
    { scope: 'punctuation.wgsl', settings: { foreground: '#969696' } },
    { scope: 'variable.other.member.wgsl', settings: { foreground: '#ababab' } },
  ],
  colors: {
    'editorGroup.border': '#343434',
    'editorGroupHeader.tabsBackground': '#0b0b0b',
    'editorGroupHeader.tabsBorder': '#242424',
    'tab.activeBackground': '#000000',
    'tab.activeForeground': '#ededed',
    'tab.inactiveBackground': '#0b0b0b',
    'tab.inactiveForeground': '#929292',
    'tab.border': '#242424',
    'tab.activeBorderTop': '#58a6ff',
    'editor.background': '#000000',
    'editorGutter.background': '#000000',
    'editor.foreground': '#c9d1d9',
    'editor.lineHighlightBackground': '#101010',
    'editor.lineHighlightBorder': '#00000000',
    'editorLineNumber.foreground': '#929292',
    'editorLineNumber.activeForeground': '#ededed',
    'editorCursor.foreground': '#58a6ff',
    'editor.selectionBackground': '#2d2d2d',
    'editor.inactiveSelectionBackground': '#232323',
    'editor.selectionHighlightBackground': '#d0d0d021',
    'editor.findMatchBackground': '#e3b34150',
    'editor.findMatchHighlightBackground': '#e3b3412b',
    'editorError.foreground': '#f85149',
    'editorWarning.foreground': '#e3b341',
    'editorInfo.foreground': '#58a6ff',
    'editorHint.foreground': '#929292',
    'editorBracketMatch.background': '#292929',
    'editorBracketMatch.border': '#858585',
    'editorBracketHighlight.foreground1': '#959595',
    'editorBracketHighlight.foreground2': '#a3a3a3',
    'editorBracketHighlight.foreground3': '#8b8b8b',
    'editorBracketHighlight.foreground4': '#999999',
    'editorBracketHighlight.foreground5': '#909090',
    'editorBracketHighlight.foreground6': '#9e9e9e',
    'editorBracketHighlight.unexpectedBracket.foreground': '#f85149',
    'editorWidget.background': '#161616',
    'editorWidget.foreground': '#c9d1d9',
    'editorWidget.border': '#343434',
    'editorHoverWidget.background': '#161616',
    'editorHoverWidget.border': '#343434',
    'editorHoverWidget.foreground': '#c9d1d9',
    'editorSuggestWidget.background': '#161616',
    'editorSuggestWidget.border': '#343434',
    'editorSuggestWidget.foreground': '#c9d1d9',
    'editorSuggestWidget.selectedBackground': '#2d2d2d',
    'editorSuggestWidget.highlightForeground': '#79c0ff',
    'menu.background': '#161616',
    'menu.foreground': '#c9d1d9',
    'menu.border': '#343434',
    'menu.separatorBackground': '#343434',
    'menu.selectionBackground': '#2d2d2d',
    'menu.selectionForeground': '#ededed',
    'list.hoverBackground': '#232323',
    'list.focusBackground': '#2d2d2d',
    'input.background': '#000000',
    'input.foreground': '#c9d1d9',
    'input.border': '#343434',
    'input.placeholderForeground': '#929292',
    'focusBorder': '#58a6ff',
    'scrollbarSlider.background': '#80808045',
    'scrollbarSlider.hoverBackground': '#8080807a',
    'scrollbarSlider.activeBackground': '#9292929c',
    'widget.shadow': '#000000b3',
  },
}

const configuration = {
  comments: { lineComment: '//', blockComment: ['/*', '*/'] },
  brackets: [['{', '}'], ['[', ']'], ['(', ')']],
  autoClosingPairs: [{ open: '{', close: '}' }, { open: '[', close: ']' }, { open: '(', close: ')' }],
}

const grammar = {
  scopeName: 'source.wgsl',
  patterns: [
    { match: '//.*$', name: 'comment.wgsl' },
    { include: '#blockComment' },
    { match: '@[a-zA-Z_]\\w*', name: 'meta.annotation.wgsl' },
    {
      match: String.raw`\b(?:0[xX][0-9a-fA-F_]+(?:\.[0-9a-fA-F_]*)?(?:[pP][+-]?\d+)?|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)[iufh]?\b|\.\d+(?:[eE][+-]?\d+)?[fh]?\b`,
      name: 'constant.numeric.wgsl',
    },
    {
      match: String.raw`(\.)([ \t]*)([a-zA-Z_]\w*)`,
      captures: { 1: { name: 'punctuation.wgsl' }, 3: { name: 'variable.other.member.wgsl' } },
    },
    {
      match: String.raw`\b(fn)\s+([a-zA-Z_]\w*)`,
      captures: { 1: { name: 'keyword.wgsl' }, 2: { name: 'entity.name.function.wgsl' } },
    },
    {
      match: String.raw`\b(struct|alias)\s+([a-zA-Z_]\w*)`,
      captures: { 1: { name: 'keyword.wgsl' }, 2: { name: 'storage.type.wgsl' } },
    },
    { match: String.raw`\b(?:true|false|[A-Z][A-Z0-9_]+)\b`, name: 'constant.language.wgsl' },
    { match: String.raw`\b(?:${wgslKeywords.join('|')})\b`, name: 'keyword.wgsl' },
    { match: String.raw`\b(?:${wgslTypes.join('|')})\b`, name: 'storage.type.wgsl' },
    { match: String.raw`\b[a-zA-Z_]\w*(?=\s*\()`, name: 'entity.name.function.wgsl' },
    { match: String.raw`\b[A-Z][a-zA-Z_0-9]*`, name: 'storage.type.wgsl' },
    { match: String.raw`[a-zA-Z_]\w*`, name: 'variable.other.identifier.wgsl' },
    { match: '[=><!~?:&|+*^%.,;/-]', name: 'punctuation.wgsl' },
  ],
  repository: {
    blockComment: {
      name: 'comment.wgsl',
      begin: String.raw`/\*`,
      end: String.raw`\*/`,
      patterns: [{ include: '#blockComment' }],
    },
  },
}

let ready: Promise<void> | undefined

export function registerWgslTheme() {
  if (ready) return ready
  const extension = registerExtension({
    name: 'wgsl-editor',
    publisher: 'earth-connect',
    version: '1.0.0',
    engines: { vscode: '*' },
    contributes: {
      languages: [{
        id: wgslLanguageId, extensions: ['.wgsl'], aliases: ['WGSL'], configuration: './language-configuration.json',
      }],
      themes: [{ id: wgslThemeName, label: wgslThemeName, uiTheme: 'vs-dark', path: './theme.json' }],
      grammars: [{ language: wgslLanguageId, scopeName: grammar.scopeName, path: './grammar.json' }],
    },
  }, ExtensionHostKind.LocalWebWorker, { system: true })
  const assets = {
    'theme.json': theme,
    'grammar.json': grammar,
    'language-configuration.json': configuration,
  }
  for (const [path, contents] of Object.entries(assets)) {
    extension.registerFileUrl(path, 'data:application/json,' + encodeURIComponent(JSON.stringify(contents)))
  }
  ready = extension.whenReady()
  return ready
}
