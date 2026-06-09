# QFlow Project Website

This is the project website for QFlow, hosted via GitHub Pages on the `gh-pages` branch.

## Structure

```
.
├── index.html           # Main website page
├── README.md           # This file
└── static/
    ├── css/
    │   └── custom.css  # Custom styling
    ├── js/
    │   └── main.js     # Interactive features
    └── images/         # Project images, figures, results
```

## Setup

The website is automatically served via GitHub Pages when you push to the `gh-pages` branch.

### Enable GitHub Pages

1. Go to your repository Settings → Pages
2. Set the source to "Deploy from a branch"
3. Select `gh-pages` as the branch
4. Save

Your site will be available at: `https://yourusername.github.io/qflow`

## Development

To view the website locally:

```bash
# Using Python 3
python -m http.server 8000

# Using Python 2
python -m SimpleHTTPServer 8000
```

Then visit `http://localhost:8000` in your browser.

## Customization

- **Content**: Edit `index.html` to update the website content
- **Styling**: Modify `static/css/custom.css` for custom styles
- **Images**: Add project figures to `static/images/`
- **Interactivity**: Extend `static/js/main.js` for more features

## Resources

- [Bulma CSS Framework](https://bulma.io) - Used for styling
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [HTML Best Practices](https://www.w3.org/html/)

## Notes

- Keep the `gh-pages` branch separate from your main code branch
- Use this branch only for website content
- Update the GitHub links in `index.html` to point to your actual repository
